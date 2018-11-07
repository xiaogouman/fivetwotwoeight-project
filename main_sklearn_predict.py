######### load csv data #########
import pandas as pd
df = pd.read_csv('./data/train_v2.csv')
df_test = pd.read_csv('./data/test_v2.csv')
df = df.rename(columns={'title': 'text'})
df_test = df_test.rename(columns={'title': 'text'})
df_combined = pd.concat([df, df_test], sort=False)


############# load training data #############
import csv, os
train_v2_dir = './articles/train_v2/'
test_v2_dir = './articles/test_v2/'


def load_train_dataset(use_title_only = True, type = 'summary', include_download_fail = False):
    x_new_col = ['article_id','text', 'publisher']
    y_column = ['category']
    if use_title_only:
        print ('use title only')
        X = df[x_new_col]
        Y = df[y_column]
        test_X = df[x_new_col]


    else:
        X, Y, test_X = [], [], []
        if type == 'summary':
            print('use article summary')
            for article_id, category, text, publisher in zip(df['article_id'], df['category'], df['text'], df['publisher']):
                if os.path.exists(train_v2_dir+'%d_summary.txt'%article_id):
                    with open(train_v2_dir+'%d_summary.txt'%article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        text = text+'\n'+next(reader)[0]
                        X.append([article_id, text, publisher])
                        Y.append(category)
                elif include_download_fail:
                    X.append([text, publisher])
                    Y.append(category)
            for article_id,text, publisher in zip(df_test['article_id'], df_test['text'], df_test['publisher']):
                if os.path.exists(test_v2_dir + '%d_summary.txt' % article_id):
                    with open(test_v2_dir + '%d_summary.txt' % article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        text = text + '\n' + next(reader)[0]
                        test_X.append([article_id, text, publisher])
                elif include_download_fail:
                    test_X.append([article_id,text, publisher])

        else:
            print('use article text')
            for article_id, category, text, publisher in zip(df['article_id'], df['category'], df['text'],
                                                              df['publisher']):
                if os.path.exists(train_v2_dir+'%d_text.txt'%article_id):
                    with open(train_v2_dir+'%d_text.txt'%article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        text = text + '\n' + next(reader)[0]
                        X.append([article_id, text, publisher])
                        Y.append(category)
                elif include_download_fail:
                    X.append([article_id, text, publisher])
                    Y.append(category)
            for article_id, text, publisher in zip(df_test['article_id'], df_test['text'],
                                                             df_test['publisher']):
                if os.path.exists(test_v2_dir+'%d_text.txt'%article_id):
                    with open(test_v2_dir+'%d_text.txt'%article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        text = text + '\n' + next(reader)[0]
                        test_X.append([article_id, text, publisher])
                elif include_download_fail:
                    test_X.append([article_id, text, publisher])
        test_X = pd.DataFrame(test_X, columns=x_new_col)
        X = pd.DataFrame(X, columns=x_new_col)
        Y = pd.DataFrame(Y, columns=y_column)
    Y = Y['category']
    print('using {0} training data'.format(len(X)))
    print(Y.value_counts())
    return X, list(Y), test_X




############## preprocess(hacking TfidfVectorizer) ################
from custom_transformers import TextTfidfVectorizer, PublisherTfidfVectorizer,NumberOfWordsExtractor, TextCountVectorizer


##################### model selection and training ##################
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, \
    confusion_matrix, roc_auc_score, make_scorer, f1_score, fbeta_score
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

#################### test data result ######################
import numpy as np

X_summary, Y_summary, X_test_summary = load_train_dataset(use_title_only=False, type='summary', include_download_fail=False)
X_title, Y_title, X_test_title = load_train_dataset(use_title_only=True, type='summary', include_download_fail=False)

def save_pred_result(name='', preds = []):
    result_dir = 'results/'
    filename = 'sampleSubmission_v2_%s.csv'%name
    y = [[i+1, preds[i]] for i in range(len(preds))]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    np.savetxt(result_dir+filename, y, header='article_id,category', delimiter=',', fmt='%d', comments='')

def predict(clf, X_train, y_train, X_test):
    clf, preds_train, preds = train_and_predict(clf, X_train, y_train, X_test)
    print('f2: {0}'.format(fbeta_score(y_train, preds_train, average='macro', beta=2)))
    print(classification_report(y_train, preds_train))
    return clf, preds

X_summary_combined = X_summary.append(X_test_summary)
X_title_combined = X_title.append(X_test_title)

vect_summary = TextCountVectorizer(ngram_range=(1, 3), max_df=.90, min_df=0.0025, max_features=2000)
vect_title = TextCountVectorizer(ngram_range=(1, 3), max_df=.90, min_df=0.001)
vect_summary.fit(X_summary_combined, y=None)
vect_title.fit(X_title_combined, y=None)

features = vect_summary.get_feature_names()
for feature in features:
    print(feature)
print('feature counts: {0}'.format(len(features)))

print('x_test count:{0}'.format(X_test_summary.count()))
X_train_summary = vect_summary.transform(X_summary)
X_test_summary = vect_summary.transform(X_test_summary)
y_train_summary = Y_summary

X_train_title = vect_title.transform(X_title)
X_test_title = vect_title.transform(X_test_title)
y_train_title = Y_title

print ('******** GradientBoostingClassifier ********')
gb = GradientBoostingClassifier()
gb, preds = predict(gb, X_train_title, y_train_title, X_test_title)
save_pred_result('gb', preds)

print ('******** AdamBoostingClassifier ********')
ada = AdaBoostClassifier()
ada, preds = predict(ada, X_train_title, y_train_title, X_test_title)
save_pred_result('ada', preds)

print ('******** XgBoostClassifier ********')
xgb = XGBClassifier()
xgb, preds = predict(xgb, X_train_title, y_train_title, X_test_title)
save_pred_result('xgb', preds)

print ('******** VotingClassifier ********')
# [2,1,2,1]
voting = VotingClassifier(estimators=[
       ('ada', ada), ('xgb', xgb), ('gb', gb)],
        weights=[1,2,3], flatten_transform=True)
voting, voting_preds = predict(voting, X_train_title, y_train_title, X_test_title)
save_pred_result('voting', voting_preds)

print('******** summary gb classifier *******')
gb_summary = GradientBoostingClassifier()
gb_summary, preds = predict(gb_summary, X_train_summary, y_train_summary, X_test_summary)
preds_index = np.vstack((list(X_summary['article_id']), preds))
print(preds_index)
# save_pred_result('gb', preds)