import pandas as pd
df = pd.read_csv('./data/train.csv')
df.head()

from io import StringIO
col = ['category', 'title']
df = df[col]
#df = df[pd.notnull(df['title'])]
df.columns = ['category', 'title']
#df['category_id'] = df['category'].factorize()[0]
category_df = df[col].drop_duplicates().sort_values('category')
category_to_id = dict(category_df.values)
df.head()

import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,6))
# df.groupby('category').category.count().plot.bar(ylim=0)
# plt.show()

#calculate Term Frequency, Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.title).toarray()
labels = df.category
print(features.shape)

# #We can use sklearn.feature_selection.chi2 to find the terms that are the most correlated with each of the products
# from sklearn.feature_selection import chi2
# import numpy as np
# N = 2
# for Product, category_id in sorted(category_to_id.items()):
#   features_chi2 = chi2(features, labels == Product)
#   indices = np.argsort(features_chi2[0])
#   feature_names = np.array(tfidf.get_feature_names())[indices]
#   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#   print("# '{}':".format(Product))
#   print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
#   print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
model = LinearSVC()
#model = MultinomialNB()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(conf_mat, annot=True, fmt='d',
#             xticklabels=category_df.category.values, yticklabels=category_df.category.values)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()
# print("finish")

from sklearn.externals import joblib
joblib.dump(model, 'model.joblib') 
print(conf_mat)
print(accuracy)