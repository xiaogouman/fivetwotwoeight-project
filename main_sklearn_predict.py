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
        test_X = df_test[x_new_col]


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


##################### model selection and training ##################
from custom_transformers import TextTfidfVectorizer, PublisherTfidfVectorizer,NumberOfWordsExtractor, TextCountVectorizer
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
from sklearn.model_selection import train_test_split
from hypopt import GridSearch

def print_scores(y_val, preds, y_train, preds_train):
    print("f2 val:", fbeta_score(y_val, preds, average='macro', beta=2))
    print("f2 train:", fbeta_score(y_train, preds_train, average='macro', beta=2))
    print(classification_report(y_val, preds))
    print(confusion_matrix(y_val, preds))

def train_and_predict(clf, X_train, y_train, X_test):
    clf = clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds_train = clf.predict(X_train)
    return clf, preds_train, preds

def f2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, average='macro', beta=2)

cross_validate = False
if cross_validate:
    X_title, Y_title, X_test_title = load_train_dataset(use_title_only=True, type='summary',
                                                        include_download_fail=False)
    for i in range(5):
        X_train, X_val, y_train, y_val = train_test_split(X_title, Y_title, test_size=0.30)
        vect_val = TextCountVectorizer(ngram_range=(1, 3), max_df=.90, min_df=0.0012)
        vect_val.fit(X_title, y=None)

        # print('training count: ', X_train.count(), 'val count: ', X_val.count())
        # features = vect_val.get_feature_names()
        # for feature in features:
        #     print(feature)
        # print('feature counts: {0}'.format(len(features)))

        X_train = vect_val.transform(X_train)
        X_val = vect_val.transform(X_val)
        print('******** GridSearch ********')
        param_grid = {
            'n_estimators': [40, 60, 80, 100, 120],
            'learning_rate': [0.1, 0.15, 0.2],
            'max_depth': [6, 7, 8, 9, 10]

        }
        scorer = make_scorer(f2)

        gs = GridSearch(model=GradientBoostingClassifier())
        gs.fit(X_train, y_train, param_grid, X_val, y_val, scoring=scorer)
        print('params: ',gs.get_best_params())
        print('Test Score for Optimized Parameters:', gs.score(X_val, y_val))

    # print('******** GradientBoostingClassifier ********')
    # gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_depth=7)
    # gb, preds_train, preds = train_and_predict(gb, X_train, y_train, X_val)
    # print_scores(y_val, preds, y_train, preds_train)
    #
    # print('******** AdamBoostingClassifier ********')
    # ada = AdaBoostClassifier()
    # ada, preds_train, preds = train_and_predict(ada, X_train, y_train, X_val)
    # print_scores(y_val, preds, y_train, preds_train)
    #
    # print('******** XgBoostClassifier ********')
    # xgb = XGBClassifier()
    # xgb, preds_train, preds = train_and_predict(xgb, X_train, y_train, X_val)
    # print_scores(y_val, preds, y_train, preds_train)
    #
    # print('******** VotingClassifier ********')
    # # [2,1,2,1]
    # voting = VotingClassifier(estimators=[
    #     ('ada', ada), ('xgb', xgb), ('gb', gb)],
    #     weights=[1, 2, 3], flatten_transform=True)
    # voting, preds_train, preds = train_and_predict(voting, X_train, y_train, X_val)
    # print_scores(y_val, preds, y_train, preds_train)

#################### test data result ######################
import numpy as np
run_test = True
if run_test:
    voting_preds = list(pd.read_csv('results/sampleSubmission_v2_voting_best.csv')['category'])
    X_title, Y_title, X_test_title = load_train_dataset(use_title_only=True, type='summary', include_download_fail=False)

    def save_pred_result(name='', preds = []):
        result_dir = 'results/'
        filename = 'sampleSubmission_v2_%s.csv'%name
        y = [[i+1, preds[i]] for i in range(len(preds))]
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        diff_counts = sum([1 for x in voting_preds-preds if x != 0])
        print('diff_counts: ', diff_counts)
        np.savetxt(result_dir+filename, y, header='article_id,category', delimiter=',', fmt='%d', comments='')

    def predict(clf, X_train, y_train, X_test):
        clf, preds_train, preds = train_and_predict(clf, X_train, y_train, X_test)
        print('f2: {0}'.format(fbeta_score(y_train, preds_train, average='macro', beta=2)))
        print(classification_report(y_train, preds_train))
        return clf, preds


    X_title_combined = X_title.append(X_test_title)
    vect_title = TextCountVectorizer(ngram_range=(1, 3), max_df=.95, min_df=.0002)
    vect_title.fit(X_title_combined, y=None)

    features = vect_title.get_feature_names()
    for feature in features:
        print(feature)
    print('feature counts: {0}'.format(len(features)))

    X_train_title = vect_title.transform(X_title)
    X_test_title = vect_title.transform(X_test_title)
    y_train_title = Y_title

    print ('******** RandomForest ********')
    rf = RandomForestClassifier(random_state=0)
    rf, preds = predict(rf, X_train_title, y_train_title, X_test_title)
    save_pred_result('rf', preds)

    print ('******** GradientBoostingClassifier ********')
    gb = GradientBoostingClassifier(random_state=0)
    gb, preds = predict(gb, X_train_title, y_train_title, X_test_title)
    save_pred_result('gb', preds)

    print ('******** AdamBoostingClassifier ********')
    ada = AdaBoostClassifier(random_state=0)
    ada, preds = predict(ada, X_train_title, y_train_title, X_test_title)
    save_pred_result('ada', preds)

    print ('******** XgBoostClassifier ********')
    xgb = XGBClassifier(random_state=0)
    xgb, preds = predict(xgb, X_train_title, y_train_title, X_test_title)
    save_pred_result('xgb', preds)

    print ('******** VotingClassifier ********')
    # [2,1,2,1]
    voting = VotingClassifier(estimators=[
           ('ada', ada), ('xgb', xgb), ('gb', gb)],
            weights=[1,2,3], flatten_transform=True)
    voting, preds = predict(voting, X_train_title, y_train_title, X_test_title)
    save_pred_result('voting', preds)


############ combined ###################
run_combined = False
if run_combined:
    X_summary, Y_summary, X_test_summary = load_train_dataset(use_title_only=False, type='summary', include_download_fail=False)
    X_summary_combined = X_summary.append(X_test_summary)

    vect_summary = TextCountVectorizer(ngram_range=(1, 3), max_df=.75, min_df=0.01)
    vect_summary.fit(X_summary_combined, y=None)
    features = vect_summary.get_feature_names()
    for feature in features:
        print(feature)
    print('feature counts: {0}'.format(len(features)))

    X_train_summary_t = vect_summary.transform(X_summary)
    X_test_summary_t = vect_summary.transform(X_test_summary)
    y_train_summary = Y_summary

    voting_preds = list(pd.read_csv('results/sampleSubmission_v2_voting_best.csv')['category'])
    def combine(X_test_partial, preds_partial, preds_all):
        preds_with_index = np.vstack((list(X_test_partial['article_id']), preds_partial)).transpose()
        pred_combined = preds_all
        count = 0
        for x in preds_with_index:
            articleId = x[0]
            previous = voting_preds[articleId - 1]
            now = x[1]
            pred_combined[articleId - 1] = now
            if previous != now:
                count += 1
                # print('previous', previous, 'now', now)
        print('changed: {0}'.format(count))
        return pred_combined

    print('******** with summary, voting - gb *******')
    gb_summary = GradientBoostingClassifier()
    gb_summary, preds = predict(gb_summary, X_train_summary_t, y_train_summary, X_test_summary_t)
    pred_combined = combine(X_test_summary, preds, voting_preds)
    save_pred_result('voting_gb', pred_combined)


    print('******** with summary, voting - ada *******')
    ada_summary = AdaBoostClassifier()
    ada_summary, preds = predict(ada_summary, X_train_summary_t, y_train_summary, X_test_summary_t)
    pred_combined = combine(X_test_summary, preds, voting_preds)
    save_pred_result('voting_ada', pred_combined)

    print('******** with summary, voting - xgb *******')
    xgb_summary = XGBClassifier()
    xgb_summary, preds = predict(xgb_summary, X_train_summary_t, y_train_summary, X_test_summary_t)
    pred_combined = combine(X_test_summary, preds, voting_preds)
    save_pred_result('voting_xgb', pred_combined)

    print('******** with summary, voting - voting *******')
    voting_summary = VotingClassifier(estimators=[
           ('ada', ada_summary), ('xgb', xgb_summary), ('gb', gb_summary)],
            weights=[1,3,4], flatten_transform=True)
    voting_summary, preds = predict(voting_summary, X_train_summary_t, y_train_summary, X_test_summary_t)
    pred_combined = combine(X_test_summary, preds, voting_preds)
    save_pred_result('voting_voting', pred_combined)

