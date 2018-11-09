######### load csv data #########
import pandas as pd
df = pd.read_csv('./data/train_v2.csv')
df_test = pd.read_csv('./data/test_v2.csv')
df = df.rename(columns={'title': 'text'})
df_test = df_test.rename(columns={'title': 'text'})


############# load training data #############
import csv, os
train_v2_dir = './articles/train_v2/'
test_v2_dir = './articles/test_v2/'

stopwords = []
with open('data/stopwords.txt', 'r') as f:
    reader = csv.reader(f)
    for r in reader:
        stopwords.append(r[0])

def load_train_dataset(use_title_only = True):
    x_new_col = ['article_id','text', 'publisher']
    y_column = ['category']
    if use_title_only:
        print ('use title only')
        X = df[x_new_col]
        Y = df[y_column]
        test_X = df_test[x_new_col]
    Y = Y['category']
    print('using {0} training data'.format(len(X)))
    print(Y.value_counts())
    return X, list(Y), test_X

##################### preprocessing ####################
import unicodedata, re
import pandas as pd
import nltk
from nltk import PorterStemmer, WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

def isfloat(str):
    try:
        s = float(str)
    except:
        return False
    return True


def remove_invalid(text):
    return re.sub('[^A-Za-z0-9]+', ' ', text)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None  # for easy if-statement


def lemmer(words):
    lemmatizer = WordNetLemmatizer()
    tagged = pos_tag(words)
    lemmed_words = []
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:  # not supply tag in case of None
            lemma = lemmatizer.lemmatize(word)
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
        lemmed_words.append(lemma)
    return lemmed_words


def stemmer(words):
    return [PorterStemmer().stem(word) for word in words]


def normalize(words):
    return [''.join((char for char in unicodedata.normalize('NFD', str(word)) if unicodedata.category(char) != 'Mn'))
            for word in words]


def is_to_remove(feature):
    # """ remove feature with all digits, or the first two items are digit"""
    words = feature.split()
    if len(words) == 1 and isfloat(words[0]):
        return True
    if len(words) > 1 and isfloat(words[1]) and isfloat(words[0]):
        return True
    return False


class TextCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        def analyser(text):
            if pd.isnull(text):
                return []
            text = text.lower()
            text = remove_invalid(text)

            #words = text.split()
            words = nltk.word_tokenize(text, language='english')
            words = normalize(words)
            words = lemmer(words)
            features = self._word_ngrams(words, self.stop_words)

            # feature selection: remove those with only one number or two numbers
            selected_features = [feature for feature in features if not is_to_remove(feature)]
            return selected_features
        return analyser

    def transform(self, X):
        X_texts = X['text']
        return super().transform(X_texts)

    def fit_transform(self, X, y=None):
        X_texts = X['text']
        return super().fit_transform(X_texts, y)


#################### test data result ######################
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, \
    confusion_matrix, roc_auc_score, make_scorer, f1_score, fbeta_score

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

run_test = True
if run_test:
    # voting_preds = list(pd.read_csv('results/sampleSubmission_v2_voting_best.csv')['category'])
    X_title, Y_title, X_test_title = load_train_dataset(use_title_only=True)

    def save_pred_result(name='', preds = []):
        filename = 'sampleSubmission_v2_%s.csv'%name
        y = [[i+1, preds[i]] for i in range(len(preds))]
        # diff_counts = sum([1 for x in voting_preds-preds if x != 0])
        # print('diff_counts: ', diff_counts)
        np.savetxt(filename, y, header='article_id,category', delimiter=',', fmt='%d', comments='')

    def predict(clf, X_train, y_train, X_test):
        clf, preds_train, preds = train_and_predict(clf, X_train, y_train, X_test)
        print('f2: {0}'.format(fbeta_score(y_train, preds_train, average='macro', beta=2)))
        print(classification_report(y_train, preds_train))
        return clf, preds


    X_title_combined = X_title.append(X_test_title)
    vect_title = TextCountVectorizer(ngram_range=(1, 3), max_df=.95, min_df=.001, stop_words=stopwords)
    vect_title.fit(X_title_combined, y=None)

    features = vect_title.get_feature_names()
    # for feature in features:
    #     print(feature)
    print('feature counts: {0}'.format(len(features)))

    X_train_title = vect_title.transform(X_title)
    X_test_title = vect_title.transform(X_test_title)
    y_train_title = Y_title

    print ('******** GradientBoostingClassifier ********')
    gb = GradientBoostingClassifier()
    gb, preds = predict(gb, X_train_title, y_train_title, X_test_title)
    # save_pred_result('gb', preds)

    print ('******** AdamBoostingClassifier ********')
    ada = AdaBoostClassifier()
    ada, preds = predict(ada, X_train_title, y_train_title, X_test_title)
    # save_pred_result('ada', preds)

    print ('******** XgBoostClassifier ********')
    xgb = XGBClassifier()
    xgb, preds = predict(xgb, X_train_title, y_train_title, X_test_title)
    # save_pred_result('xgb', preds)

    print ('******** VotingClassifier ********')
    voting = VotingClassifier(estimators=[
           ('ada', ada), ('xgb', xgb), ('gb', gb)],
            weights=[1,2,3], flatten_transform=True)
    voting, preds = predict(voting, X_train_title, y_train_title, X_test_title)
    save_pred_result('voting', preds)


