######### load csv data #########
import pandas as pd
df = pd.read_csv('./data/train_v2.csv')
df = df.rename({'title':'text'}, axis=1)
df_test = pd.read_csv('./data/test_v2.csv')
# 4    2726
# 2    1701
# 0     798
# 3     453
# 1     349

# map publisher
publishers = pd.concat((df, df_test), sort=True).drop_duplicates('publisher')['publisher']
index_to_publisher = publishers.to_dict()
publisher_to_index = {index_to_publisher[k]: k for k in index_to_publisher}
df['publisher'] = df['publisher'].apply(lambda x: x.map(publisher_to_index.get(x)))


# train class 0,1,2,3 first?
# df = df[df['category'] < 4]


# binary classification between 0,1,2,3 and 4 first?
# Y = [1 if y==4 else 0 for y in Y]

############# load training data #############
import csv, os
train_v2_dir = './articles/train_v2/'


def load_train_dataset(use_title_only = True, type = 'summary', include_download_fail = False):
    if use_title_only:
        print ('use title only')
        X = df[['text', 'publisher']]
        Y = list(df['category'])

    else:
        X = []
        Y = []
        if type == 'summary':
            print('use article summary')
            for article_id, category, title in zip(df['article_id'], df['category'], df['title']):
                if os.path.exists(train_v2_dir+'%d_summary.txt'%article_id):
                    with open(train_v2_dir+'%d_summary.txt'%article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        X.append(title+'\n'+next(reader)[0])
                        Y.append(category)
                elif include_download_fail:
                    X.append(title)
                    Y.append(category)
        else:
            print('use article text')
            for article_id, category, title in zip(df['article_id'], df['category'], df['title']):
                if os.path.exists(train_v2_dir+'%d_text.txt'%article_id):
                    with open(train_v2_dir+'%d_text.txt'%article_id, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        X.append(title+'\n'+next(reader)[0])
                        Y.append(category)
                elif include_download_fail:
                    X.append(title)
                    Y.append(category)
    print('using {0} training data'.format(len(X)))
    for i in range(0, 5):
        print('class {0}: {1}'.format(i, Y.count(i)))
    return X, Y


from sklearn.model_selection import train_test_split
X, Y = load_train_dataset(use_title_only=True, type='summary', include_download_fail=False)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)


############## preprocess(hacking TfidfVectorizer) ################
stopwords = []
with open('stopwords.txt', 'r') as f:
    reader = csv.reader(f)
    for r in reader:
        stopwords.append(r[0])


import unicodedata, re
from nltk import PorterStemmer, WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
class MyTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        def remove_invalid(text):
            return re.sub(u'[•₹€–—�►©!“”"’✓#$%&\'()*+,-./:;<=>?@，。?★、…[\\]^_`{|}~]+', ' ', text)

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

        def is_all_digit(feature):
            words = feature.split(' ')
            for word in words:
                if not word.isdigit():
                    return False
            return True

        def analyser(text):
            text = text.lower()
            text = remove_invalid(text)

            words = text.split()
            words = normalize(words)
            words = lemmer(words)
            features = self._word_ngrams(words, stopwords)

            # feature selection: remove those with only one number or two numbers
            selected_features = [feature for feature in features if not is_all_digit(feature)]
            return selected_features
        return analyser

    def transform(self, X):
        X_texts = X['text']
        X_texts = super(TfidfVectorizer, self).transform(X_texts)
        return self._tfidf.transform(X_texts, copy=False)

    def fit(self, X, y):
        X_texts = X['text']
        self._check_params()
        X_texts = super(TfidfVectorizer, self).fit_transform(X_texts)
        self._tfidf.fit(X_texts)
        return self

    def fit_transform(self, X, y=None):
        X_texts = X['text']
        self._check_params()
        X_texts = super(TfidfVectorizer, self).fit_transform(X_texts)
        self._tfidf.fit(X_texts)
        return self._tfidf.transform(X_texts, copy=False)



vect = MyTfidfVectorizer(ngram_range=(1,3), max_df=.95, use_idf=True, norm='l2')
X_train_tdidf = vect.fit_transform(X_train, y_train)
X_test_tdidf = vect.transform(X_test)

features = vect.get_feature_names()
print('feature counts: {0}'.format(len(features)))
# for feature in features:
#     print(feature)

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
# pipe = Pipeline([
#     ('features', FeatureUnion([
#         ('tf_idf', MyTfidfVectorizer(ngram_range=(1,3), max_df=.95, use_idf=True, norm='l2')),
#         ('text_length', Map)
#     ])),
#     ('clf', LinearSVC())
# ])

##################### model selection and training ##################

from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, roc_auc_score


def print_scores(y_test, preds, y_train, preds_train):
    print("Accuracy test:", accuracy_score(y_test, preds))
    print("Accuracy train:", accuracy_score(y_train, preds_train))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

run = True
if run:
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
print ('****************** imbalance learn ****************')
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
print ('************** RandomUnderSampler ***********')
pipe = make_pipeline_imb(RandomUnderSampler(),
                         RandomForestClassifier())
pipe.fit(X_train_tdidf, y_train)
preds = pipe.predict(X_test_tdidf)
preds_train = pipe.predict(X_train_tdidf)
print(classification_report_imbalanced(y_test, preds))

print ('************** RandomOverSampler ***********')
pipe = make_pipeline_imb(RandomOverSampler(),
                         RandomForestClassifier())
pipe.fit(X_train_tdidf, y_train)
preds = pipe.predict(X_test_tdidf)
preds_train = pipe.predict(X_train_tdidf)
print(classification_report_imbalanced(y_test, preds))

print ('************** SMOTEENN(combine) ***********')
pipe = make_pipeline_imb(SMOTEENN(random_state=42),
                         RandomForestClassifier())
pipe.fit(X_train_tdidf, y_train)
preds = pipe.predict(X_test_tdidf)
preds_train = pipe.predict(X_train_tdidf)
print(classification_report_imbalanced(y_test, preds))

print ('************** BalancedRandomForestClassifier(ensemble) ***********')
pipe = make_pipeline_imb(BalancedRandomForestClassifier(max_depth=40))
pipe.fit(X_train_tdidf, y_train)
preds = pipe.predict(X_test_tdidf)
preds_train = pipe.predict(X_train_tdidf)
print(classification_report_imbalanced(y_test, preds))

print ('************** BalancedBaggingClassifier(ensemble) ***********')
pipe = make_pipeline_imb(CountVectorizer(),
                         BalancedBaggingClassifier(random_state=42))
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
preds_train = pipe.predict(X_train)
print(classification_report_imbalanced(y_test, preds))