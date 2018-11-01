import unicodedata, re, csv
import pandas as pd
import numpy as np
import nltk
from nltk import PorterStemmer, WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


stopwords = []
with open('stopwords.txt', 'r') as f:
    reader = csv.reader(f)
    for r in reader:
        stopwords.append(r[0])


class TextTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        def remove_invalid(text):
            #return re.sub(u'[×•₹€–—�►©!“”"’✓#$%&\'()*+,-./:;<=>?@，。?★、…[\\]^_`{|}~]+', ' ', text)
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
            """ remove feature with all digits, or the first two items are digit"""
            words = feature.split()
            count = 0
            if len(words) == 1 and words[0].isdigit():
                return True
            if len(words) > 1 and words[1].isdigit() and words[0].isdigit():
                return True
            return False

        def analyser(text):
            if pd.isnull(text):
                return []
            text = text.lower()
            text = remove_invalid(text)

            #words = text.split()
            words = nltk.word_tokenize(text, language='english')
            words = normalize(words)
            words = lemmer(words)
            features = self._word_ngrams(words, stopwords)

            # feature selection: remove those with only one number or two numbers
            selected_features = [feature for feature in features if not is_to_remove(feature)]
            return selected_features
        return analyser

    def transform(self, X):
        X_texts = X['text']
        return super().transform(X_texts)

    def fit(self, X, y):
        X_texts = X['text']
        return super().fit(X_texts)

    def fit_transform(self, X, y=None):
        X_texts = X['text']
        return super().fit_transform(X_texts, y)

class PublisherTfidfVectorizer(TfidfVectorizer):
    def transform(self, X):
        X_texts = X['publisher'].apply(lambda x: str(x))
        return super().transform(X_texts)

    def fit(self, X, y):
        X_texts = X['publisher'].apply(lambda x: str(x))
        return super().fit(X_texts)

    def fit_transform(self, X, y=None):
        X_texts = X['publisher'].apply(lambda x: str(x))
        return super().fit_transform(X_texts, y)


from sklearn.base import BaseEstimator, TransformerMixin
class PublisherExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, vars):
        self.vars = vars

    def transform(self, X, y=None):
        return X[[self.vars]]

    def fit(self, X, y=None):
        return self


class NumberOfWordsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, vars):
        self.vars = vars  # e.g. pass in a column name to extract

    def transform(self, X, y=None):
        texts = X[self.vars]
        total_word_counts = []
        number_occ = []
        for text in texts:
            total_word_counts.append(len(text.split()))
            number_occ.append(sum([word.isdigit() for word in text.split()]))
        return np.atleast_2d([total_word_counts, number_occ]).T

    def fit(self, X, y=None):
        return self  # generally does nothing


#################### test lemmer ######################
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

def get_wordnet_pos(treebank_tag):
    print (treebank_tag)
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

def stemmer(words):
    return [PorterStemmer().stem(word) for word in words]

if __name__ == '__main__':
    print(lemmer(['weakest', 'weaker', 'weakness', 'mixed', 'series']))
    print(stemmer(['weakest', 'weaker', 'weakness', 'mixed', 'series']))