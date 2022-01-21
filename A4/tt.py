import sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import stop_words
from nltk.tokenize import TreebankWordTokenizer

import numpy as np
import pandas as pd
import pprint

# from google.colab import drive
# drive.mount('/content/gdrive')

# with open('/content/gdrive/My Drive/437-A4/SMSSpamCollection', 'r') as f: 
#     # print(f.read())
#     df = pd.read_csv(f, sep = "\t", header=None, names=["messagetype", "messagedata"])
# # categories = ['messagetype', 'messagedata']
# # X = dataf.loc[:, categories]
df = pd.read_csv('SMSSpamCollection', sep = "\t", header=None, names=["messagetype", "messagedata"])
y_train = df['messagetype']
x_train = df.drop('messagetype',axis=1)
# print(dataf)

# print('List of newsgroup categories:')
# print(list(dataf.head()))

count_vect = CountVectorizer()

tokenizer = TreebankWordTokenizer()
count_vect.set_params(tokenizer=tokenizer.tokenize)

count_vect.set_params(stop_words='english')
print(stop_words.ENGLISH_STOP_WORDS)

count_vect.set_params(ngram_range=(1,2))

count_vect.set_params(max_df=0.5)

count_vect.set_params(min_df=2)

# X_counts = count_vect.fit_transform(x_train)

# tfidf_transformer = TfidfTransformer()
# X_tfidf = tfidf_transformer.fit_transform(X_counts)

# clf = MultinomialNB().fit(X_tfidf, y_train)
# scores = cross_val_score(clf, X_tfidf, y_train, cv=3, \
#                          scoring='accuracy')
# print('Number of documents:', len(y_train), ', 3-fold accuracy:', np.mean(scores))
