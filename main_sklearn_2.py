######### load csv data #########
import pandas as pd
df = pd.read_csv('./data/train_v2.csv')
df_test = pd.read_csv('./data/test_v2.csv')
df = df.rename(columns={'title': 'text'})
df_test = df_test.rename(columns={'title': 'text'})
df_combined = pd.concat([df, df_test], sort=False)
# 4    2726
# 2    1701
# 0     798
# 3     453
# 1     349

# map publisher
# publishers = pd.concat((df, df_test), sort=True).drop_duplicates('publisher')['publisher']
# index_to_publisher = publishers.to_dict()
# publisher_to_index = {index_to_publisher[k]: k for k in index_to_publisher}
# df['publisher'] = df['publisher'].apply(lambda x: publisher_to_index.get(x))


# train class 0,1,2,3 first?
# df = df[df['category'] < 4]


# binary classification between 0,1,2,3 and 4 first?
# Y = [1 if y==4 else 0 for y in Y]

############# load training data #############
import csv, os
train_v2_dir = './articles/train_v2/'
test_v2_dir = './articles/test_v2/'


def load_train_dataset(use_title_only = True, type = 'summary', include_download_fail = False):
    x_new_col = ['text', 'publisher']
    y_column = ['category']
    if use_title_only:
        print ('use title only')
        X = df[x_new_col]
        Y = df[y_column]


    else:
        X, Y = [], []
        if type == 'summary':
            print('use article summary')
            for article_id, category, text, publisher in zip(df['article_id'], df['category'], df['text'], df['publisher']):
                if os.path.exists(train_v2_dir+'%d_summary.txt'%article_id):
                    with open(train_v2_dir+'%d_summary.txt'%article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        text = text+'\n'+next(reader)[0]
                        X.append([text, publisher])
                        Y.append(category)
                elif include_download_fail:
                    X.append([text, publisher])
                    Y.append(category)

        else:
            print('use article text')
            for article_id, category, text, publisher in zip(df['article_id'], df['category'], df['text'],
                                                              df['publisher']):
                if os.path.exists(train_v2_dir+'%d_text.txt'%article_id):
                    with open(train_v2_dir+'%d_text.txt'%article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        text = text + '\n' + next(reader)[0]
                        X.append([text, publisher])
                        Y.append(category)
                elif include_download_fail:
                    X.append([text, publisher])
                    Y.append(category)
        X = pd.DataFrame(X, columns=x_new_col)
        Y = pd.DataFrame(Y, columns=y_column)
    Y = Y['category']
    print('using {0} training data'.format(len(X)))
    print(Y.value_counts())
    return X, list(Y)


from sklearn.model_selection import train_test_split
X_train_v2, Y_train_v2 = load_train_dataset(use_title_only=True, type='summary', include_download_fail=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_v2, Y_train_v2, test_size=0.30)


############## preprocess(hacking TfidfVectorizer) ################
from custom_transformers import TextTfidfVectorizer, PublisherTfidfVectorizer,NumberOfWordsExtractor, TextCountVectorizer

def get_vect(X, y=None):
    ############ tfidf
    # vect = TextTfidfVectorizer(ngram_range=(1,3), max_df=.95, min_df=.0025, use_idf=True, norm='l2')

    ############ bow
    vect = TextCountVectorizer(ngram_range=(1, 3), max_df=.95, min_df=0.0024)

    ########### fit all data
    vect.fit(X, y=None)

    ########### fit training data
    # vect.fit(X_train, y=None)

    features = vect.get_feature_names()
    for feature in features:
        print(feature)
    print('feature counts: {0}'.format(len(features)))
    return vect


vect = get_vect(X_train_v2, Y_train_v2)

##################### model selection and training ##################

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, roc_auc_score, make_scorer, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier


def print_scores(y_val, preds, y_train, preds_train):
    print("f1 val:", f1_score(y_val, preds, average='micro'))
    print("f1 train:", f1_score(y_train, preds_train, average='micro'))
    print(classification_report(y_val, preds))
    print(confusion_matrix(y_val, preds))

def train_and_predict(clf, X_train, y_train, X_test):
    clf = clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds_train = clf.predict(X_train)
    return clf, preds_train, preds


run_gridcv = False
if run_gridcv:
    params = [{
        # 'features__transformer_weights': [[1, 0.2, 0.2],[1, 1, 1],[1, 0.5, 0.2]],
        'features__text__ngram_range':[(1,2), (1,3), (1,4)],
        'features__text__max_df':[0.95, 0.9, 0.85],
        'features__text__max_features': [1000, 3000, 5000, 10000, 20000],
        'features__publisher__ngram_range': [(1, 1), (1, 2)],
        'features__publisher__max_df': [0.95, 0.9, 0.85],
        'gbc__n_estimators':[10, 50, 100]

    }]

    scoring = {'Accuracy': make_scorer(accuracy_score), 'F1': 'f1_micro'}

    pipe = Pipeline([
        ('features', FeatureUnion([
            ('text', vect),
            ('word_counts', NumberOfWordsExtractor('text'))
        ], transformer_weights={'text': 1, 'word_counts': 0.2})),
        ('gbc', GradientBoostingClassifier())
    ])

    grid_search = pipe
    grid_search.fit(X_train, y_train)

    # grid_search = GridSearchCV(pipe, params, n_jobs=-1, verbose=1, scoring='f1', cv=5)
    # grid_search.fit(X_train, y_train)
    # print ('Best score: %0.3f' % grid_search.best_score_)
    # print ('Best parameters set:')
    # best_parameters = grid_search.best_estimator_.get_params()
    # print (best_parameters)

    preds = grid_search.predict(X_val)
    preds_train = grid_search.predict(X_train)
    print_scores(y_val, preds, y_train, preds_train)


run = False
if run:
    X_train_tdidf = vect.transform(X_train)
    X_val_tdidf = vect.transform(X_val)

    print('******** linear SVC ********')
    linear_svc = LinearSVC()
    linear_svc, preds_train, preds = train_and_predict(linear_svc, X_train_tdidf, y_train, X_val_tdidf)
    print_scores(y_val, preds, y_train, preds_train)

    print ('******** Balanced linear SVC ********')
    svc_balance = SVC(kernel='linear', probability=True, class_weight='balanced')
    svc_balance, preds_train, preds = train_and_predict(svc_balance, X_train_tdidf, y_train, X_val_tdidf)
    print_scores(y_val, preds, y_train, preds_train)

    print ('******** MultinomialNB ********')
    multinomial_nb = MultinomialNB()
    multinomial_nb, preds_train, preds = train_and_predict(multinomial_nb, X_train_tdidf, y_train, X_val_tdidf)
    print_scores(y_val, preds, y_train, preds_train)

    print ('******** RandomForest ********')
    rf = RandomForestClassifier(n_estimators=40)
    rf, preds_train, preds = train_and_predict(rf, X_train_tdidf, y_train, X_val_tdidf)
    print_scores(y_val, preds, y_train, preds_train)

    print ('******** Balanced RandomForest ********')
    rf_balance = RandomForestClassifier(class_weight='balanced')
    rf_balance, preds_train, preds = train_and_predict(rf_balance, X_train_tdidf, y_train, X_val_tdidf)
    print_scores(y_val, preds, y_train, preds_train)

    print ('******** GradientBoostingClassifier ********')
    gb = GradientBoostingClassifier()
    gb, preds_train, preds = train_and_predict(gb, X_train_tdidf, y_train, X_val_tdidf)
    print_scores(y_val, preds, y_train, preds_train)

    print ('******** VotingClassifier ********')
    voting = VotingClassifier(estimators=[
           ('linear_svc', linear_svc), ('multinomial_nb', multinomial_nb), ('rf', rf), ('gb', gb)],
            weights=[2,1,2,1], flatten_transform=True)
    voting, preds_train, preds = train_and_predict(voting, X_train_tdidf, y_train, X_val_tdidf)
    print_scores(y_val, preds, y_train, preds_train)


# https://elitedatascience.com/imbalanced-classes
# https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets


####################### imbalance learn ####################################
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
imb_run = False
if imb_run:
    print('****************** imbalance learn ****************')

    clf = RandomForestClassifier()

    print('************** RandomUnderSampler ***********')
    pipe = make_pipeline_imb(vect,
                             RandomUnderSampler(random_state=777),
                             clf)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_val, preds))

    print('************** RandomOverSampler ***********')
    pipe = make_pipeline_imb(vect,
                             RandomOverSampler(random_state=777),
                             clf)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_val, preds))

    print('************** SMOTEENN(combine) ***********')
    pipe = make_pipeline_imb(vect,
                             SMOTEENN(random_state=42),
                             clf)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_val, preds))

    print('************** BalancedRandomForestClassifier(ensemble) ***********')
    pipe = make_pipeline_imb(vect,
                             BalancedRandomForestClassifier(max_depth=40))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_val, preds))

    print('************** BalancedBaggingClassifier(ensemble) ***********')
    pipe = make_pipeline_imb(vect,
                             BalancedBaggingClassifier(random_state=42))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_val, preds))


#################### test data result ######################
import numpy as np
def save_pred_result(name='', preds = []):
    result_dir = 'results/'
    filename = 'sampleSubmission_v2_%s.csv'%name
    y = [[i+1, preds[i]] for i in range(len(preds))]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    np.savetxt(result_dir+filename, y, header='article_id,category', delimiter=',', fmt='%d', comments='')

X_combined = df_combined[['text']]
X_test = df_test[['text']]

vect = TextCountVectorizer(ngram_range=(1, 3), max_df=.95, min_df=0.001)
vect.fit(X_combined, y=None)
features = vect.get_feature_names()
for feature in features:
    print(feature)
print('feature counts: {0}'.format(len(features)))

X_train = vect.transform(X_train_v2)
X_test = vect.transform(X_test)
y_train = Y_train_v2

# imbalance learn resample
# sampler = RandomOverSampler(random_state=777)
# X_train, y_train = sampler.fit_resample(X_train, y_train)

print('******** linear SVC ********')
linear_svc = LinearSVC()
linear_svc, preds_train, preds = train_and_predict(linear_svc, X_train, y_train, X_test)
print(classification_report(y_train, preds_train))
save_pred_result('linear_svc', preds)

print ('******** Balanced linear SVC ********')
svc_balance = SVC(kernel='linear', probability=True, class_weight='balanced')
svc_balance, preds_train, preds = train_and_predict(svc_balance, X_train, y_train, X_test)
print(classification_report(y_train, preds_train))
save_pred_result('linear_svc_balance', preds)

print ('******** MultinomialNB ********')
multinomial_nb = MultinomialNB()
multinomial_nb, preds_train, preds = train_and_predict(multinomial_nb, X_train, y_train, X_test)
print(classification_report(y_train, preds_train))
save_pred_result('nb', preds)

print ('******** RandomForest ********')
rf = RandomForestClassifier(n_estimators=40)
rf, preds_train, preds = train_and_predict(rf, X_train, y_train, X_test)
print(classification_report(y_train, preds_train))
save_pred_result('rf', preds)

print ('******** Balanced RandomForest ********')
rf_balance = RandomForestClassifier(class_weight='balanced')
rf_balance, preds_train, preds = train_and_predict(rf_balance, X_train, y_train, X_test)
print(classification_report(y_train, preds_train))
save_pred_result('rf_balance', preds)

print ('******** GradientBoostingClassifier ********')
gb = GradientBoostingClassifier()
gb, preds_train, preds = train_and_predict(gb, X_train, y_train, X_test)
print(classification_report(y_train, preds_train))
save_pred_result('gb', preds)

print ('******** VotingClassifier ********')
# [2,1,2,1]
voting = VotingClassifier(estimators=[
       ('linear_svc', linear_svc), ('multinomial_nb', multinomial_nb), ('rf', rf), ('gb', gb)],
        weights=[1,1,1,2], flatten_transform=True)
voting, preds_train, preds = train_and_predict(voting, X_train, y_train, X_test)
print(classification_report(y_train, preds_train))
save_pred_result('voting', preds)