import pandas as pd
# df = pd.read_csv('./data/train_v2.csv')
df = pd.read_csv('./data/test_v2.csv')

article_ids = df['article_id']
urls = df['url']

import os
# directory = './articles/train_v2/'
directory = './articles/test_v2/'

if not os.path.exists(directory):
    os.makedirs(directory)

import csv
import newspaper
from newspaper import Article, Config
import time


# increase timeout
config = Config()
config.request_timeout = 50
# if request is not successful for 10 seconds, give up
max_wait = 10
for article_id, url in zip(article_ids, urls):
    if os.path.isfile(directory + '%d_summary.txt' % article_id) and os.path.isfile(directory + '%d_text.txt' % article_id):
        continue
    try:
        url = url.replace('\\', '/')
        article = Article(url, config=config)
        article.download()
        wait = 0
        while article.download_state == 0 and wait < max_wait:  # ArticleDownloadState.NOT_STARTED is 0
            time.sleep(1)
            wait += 1

        article.parse()
        article.nlp()
        # save summary and text
        with open(directory + '%d_summary.txt' % article_id, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([article.summary])
        with open(directory + '%d_text.txt' % article_id, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([article.text])

    except newspaper.article.ArticleException as e:
        print('download fails: article id = {0}, {1}'.format(article_id, e))

print('finish!')
