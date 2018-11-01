from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]
        
class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]

import nltk
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words

from textblob import TextBlob

# Use TextBlob
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

import pandas as pd
df = pd.read_csv('./data/train.csv')
df.head()

X = df['title']
Y = df['category']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

classifier = Pipeline([
    ('features', FeatureUnion([
        ('title', Pipeline([
            # ('colext', TextSelector('title')),
            ('tfidf', TfidfVectorizer(tokenizer=textblob_tokenizer, stop_words='english',
                     min_df=.0025, max_df=0.25, ngram_range=(1,3))),
            ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
        ])),
        # ('words', Pipeline([
        #     ('wordext', NumberSelector('TotalWords')),
        #     ('wscaler', StandardScaler()),
        # ])),
    ])),
    ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
#    ('clf', RandomForestClassifier()),
    ])

classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
print ("Accuracy:", accuracy_score(y_test, preds))
#print ("Precision:", precision_score(y_test, preds))
print (classification_report(y_test, preds))
print (confusion_matrix(y_test, preds))