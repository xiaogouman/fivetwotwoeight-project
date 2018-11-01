import pandas as pd
df = pd.read_csv('./data/train_v2.csv')
df.head()

X = df['title']
Y = df['category']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

import bow
import os
import numpy as np
import csv
from types import SimpleNamespace
if not os.path.exists('doc_classes/'):
    os.makedirs('doc_classes/')

stopwords = []
with open('stopwords.txt', 'r') as f:
    reader = csv.reader(f)
    for r in reader:
        stopwords.append(r[0])
print (stopwords.index('us'))

def preprocess(text):
    words = text.split()
    return ' '.join([word for word in words if word not in stopwords])

titles = [[] for i in range(5)]
for title, cat_id in zip(X_train, y_train):
    titles[cat_id].append(preprocess(title))

for cat_id, titles_by_cat in enumerate(titles):
    with open('doc_classes/cat_%d.txt'%cat_id, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for title in titles_by_cat:
            writer.writerow([title])

#
# arg = {}
# arg['filter'] = 'text'
# arg['lang_filter'] = 'english'
# arg['stemming_filter'] = 5
# for i in range(0 ,5):
#     arg['filename'] = 'cat_%d'%i
#     n = SimpleNamespace(**arg)
#     bow._create(n)
#
# learn_arg = {}
# learn_arg['url'] = False
# learn_arg['dir'] = False
# learn_arg['zip'] = False
# learn_arg['no_learn'] = False
# learn_arg['list_top_words'] = 50
# learn_arg['rewrite'] = False
# for i in range(0, 5):
#     learn_arg['filename'] = 'cat_%d'%i
#     learn_arg['file'] = ['doc_classes/cat_%d.txt'%i]
#     n = SimpleNamespace(**learn_arg)
#     bow._learn(n)
#
#
# cat_0 = bow.Document.load('cat_0')
# cat_1 = bow.Document.load('cat_1')
# cat_2 = bow.Document.load('cat_2')
# cat_3 = bow.Document.load('cat_3')
# cat_4 = bow.Document.load('cat_4')

dc = bow.DefaultDocument()
dc.read_text('"Visa, MasterCard ties 2 Russian banks playing play"')

preds = []
for title in X_test:
    dc = bow.DefaultDocument()
    dc.read_text(title)
    result = bow.document_classifier(dc, cat_0 = cat_0, cat_1=cat_1, cat_2=cat_2, cat_3=cat_3, cat_4=cat_4)
    preds.append(int(result[0][0].split('_')[1]))

train_preds = []
for title in X_train:
    dc = bow.DefaultDocument()
    dc.read_text(title)
    result = bow.document_classifier(dc, cat_0 = cat_0, cat_1=cat_1, cat_2=cat_2, cat_3=cat_3, cat_4=cat_4)
    train_preds.append(int(result[0][0].split('_')[1]))

from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, f1_score
print ("Accuracy test:", accuracy_score(y_test, preds))
print ("Accuracy train:", accuracy_score(y_train, train_preds))
print (classification_report(y_test, preds))
print (confusion_matrix(y_test, preds))





