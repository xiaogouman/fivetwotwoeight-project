######### load csv data #########
import pandas as pd
import numpy as np
df = pd.read_csv('./data/train_v2.csv')
df_test = pd.read_csv('./data/test_v2.csv')
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


def load_train_dataset(use_title_only = True, type = 'summary', include_download_fail = False):
    x_columns = ['text', 'publisher']
    y_column = ['category']
    if use_title_only:
        print ('use title only')
        X = df[['title', 'publisher']]
        X.columns = x_columns
        Y = df[y_column]

    else:
        X, Y = [], []
        if type == 'summary':
            print('use article summary')
            for article_id, category, title, publisher in zip(df['article_id'], df['category'], df['title'], df['publisher']):
                if os.path.exists(train_v2_dir+'%d_summary.txt'%article_id):
                    with open(train_v2_dir+'%d_summary.txt'%article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        text = title+'\n'+next(reader)[0]
                        X.append([text, publisher])
                        Y.append(category)
                elif include_download_fail:
                    text = title
                    X.append([text, publisher])
                    Y.append(category)

        else:
            print('use article text')
            for article_id, category, title, publisher in zip(df['article_id'], df['category'], df['title'],
                                                              df['publisher']):
                if os.path.exists(train_v2_dir+'%d_text.txt'%article_id):
                    with open(train_v2_dir+'%d_text.txt'%article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        text = title + '\n' + next(reader)[0]
                        X.append([text, publisher])
                        Y.append(category)
                elif include_download_fail:
                    text = title
                    X.append([text, publisher])
                    Y.append(category)
        X = pd.DataFrame(X, columns=x_columns)
        Y = pd.DataFrame(Y, columns=y_column)
    Y = Y['category']
    print('using {0} training data'.format(len(X)))
    print(Y.value_counts())
    return X, list(Y)


from sklearn.model_selection import train_test_split
X, Y = load_train_dataset(use_title_only=False, type='text', include_download_fail=True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)


############## preprocess(hacking TfidfVectorizer) ################
from custom_transformers import TextTfidfVectorizer, PublisherTfidfVectorizer,NumberOfWordsExtractor

vect = TextTfidfVectorizer(ngram_range=(1,3), max_df=.95, min_df=.0025, use_idf=True, norm='l2')

########### fit all data
vect.fit(X, y=None)

########### fit training data
#vect.fit(X_train, y=None)

features = vect.get_feature_names()
for feature in features:
    print(feature)
print('feature counts: {0}'.format(len(features)))

##################### model selection and training ##################

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, roc_auc_score, make_scorer, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier

def print_scores(y_test, preds, y_train, preds_train):
    print("Accuracy test:", accuracy_score(y_test, preds))
    print("Accuracy train:", accuracy_score(y_train, preds_train))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

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
            ('publisher', PublisherTfidfVectorizer(ngram_range=(1, 1), max_df=.95, use_idf=True, norm='l2')),
            ('word_counts', NumberOfWordsExtractor('text'))
        ], transformer_weights=None)),
        ('gbc', GradientBoostingClassifier())
    ])

    # grid_search = pipe
    # grid_search.fit(X_train, y_train)

    grid_search = GridSearchCV(pipe, params, n_jobs=-1, verbose=1, scoring=scoring, refit='F1', cv=5)
    grid_search.fit(X_train, y_train)
    print ('Best score: %0.3f' % grid_search.best_score_)
    print ('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    print (best_parameters)

    preds = grid_search.predict(X_test)
    preds_train = grid_search.predict(X_train)
    print_scores(y_test, preds, y_train, preds_train)


run = True
if run:
    X_train_tdidf = vect.transform(X_train)
    X_test_tdidf = vect.transform(X_test)

    print ('******** linear SVC ********')
    clf1 = SVC(kernel='linear', probability=True)
    clf1 = clf1.fit(X_train_tdidf, y_train)
    preds = clf1.predict(X_test_tdidf)
    preds_train = clf1.predict(X_train_tdidf)

    print_scores(y_test, preds, y_train, preds_train)

    print ('******** Balanced linear SVC ********')
    clf5 = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf5 = clf5.fit(X_train_tdidf, y_train)
    preds = clf5.predict(X_test_tdidf)
    preds_train = clf5.predict(X_train_tdidf)

    print_scores(y_test, preds, y_train, preds_train)

    print ('******** MultinomialNB ********')
    clf2 = MultinomialNB()
    clf2 = clf2.fit(X_train_tdidf, y_train)
    preds = clf2.predict(X_test_tdidf)
    preds_train = clf2.predict(X_train_tdidf)

    print_scores(y_test, preds, y_train, preds_train)


    print ('******** RandomForest ********')
    clf3 = RandomForestClassifier(n_estimators=40)
    clf3 = clf3.fit(X_train_tdidf, y_train)
    preds = clf3.predict(X_test_tdidf)
    preds_train = clf3.predict(X_train_tdidf)

    print_scores(y_test, preds, y_train, preds_train)

    print ('******** Balanced RandomForest ********')
    clf6 = RandomForestClassifier(class_weight='balanced')
    clf6 = clf6.fit(X_train_tdidf, y_train)
    preds = clf6.predict(X_test_tdidf)
    preds_train = clf6.predict(X_train_tdidf)

    print_scores(y_test, preds, y_train, preds_train)


    print ('******** GradientBoostingClassifier ********')
    clf4 = GradientBoostingClassifier()
    clf4 = clf4.fit(X_train_tdidf, y_train)
    preds = clf4.predict(X_test_tdidf)
    preds_train = clf4.predict(X_train_tdidf)

    print_scores(y_test, preds, y_train, preds_train)


    print ('******** VotingClassifier ********')
    eclf = VotingClassifier(estimators=[
           ('linearsvc', clf1), ('nb', clf2), ('rf', clf3), ('gb', clf4)],
            weights=[2,1,2,1],
           flatten_transform=True)
    eclf = eclf.fit(X_train_tdidf, y_train)
    preds = eclf.predict(X_test_tdidf)
    preds_train = eclf.predict(X_train_tdidf)

    print_scores(y_test, preds, y_train, preds_train)


# https://elitedatascience.com/imbalanced-classes
# https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets


####################### imbalance learn ####################################
imb_run = True
if imb_run:
    print ('****************** imbalance learn ****************')
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.combine import SMOTEENN
    from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
    from imblearn.pipeline import make_pipeline as make_pipeline_imb
    from imblearn.metrics import classification_report_imbalanced
    clf = RandomForestClassifier()
    print ('************** RandomUnderSampler ***********')
    pipe = make_pipeline_imb(vect,
                             RandomUnderSampler(random_state=777),
                             clf)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_test, preds))

    print ('************** RandomOverSampler ***********')
    pipe = make_pipeline_imb(vect,
                             RandomOverSampler(random_state=777),
                             clf)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_test, preds))

    print ('************** SMOTEENN(combine) ***********')
    pipe = make_pipeline_imb(vect,
                             SMOTEENN(random_state=42),
                             clf)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_test, preds))

    print ('************** BalancedRandomForestClassifier(ensemble) ***********')
    pipe = make_pipeline_imb(vect,
                             BalancedRandomForestClassifier(max_depth=40))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_test, preds))

    print ('************** BalancedBaggingClassifier(ensemble) ***********')
    pipe = make_pipeline_imb(vect,
                             BalancedBaggingClassifier(random_state=42))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    preds_train = pipe.predict(X_train)
    print(classification_report_imbalanced(y_test, preds))