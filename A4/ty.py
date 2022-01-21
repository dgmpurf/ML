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

df = pd.read_csv('SMSSpamCollection.txt', sep = "\t", header=None, names=["mtypes", "ms"])

categories = ['mtypes','ms']
newsgroups = df(categories=categories)

print('List of newsgroup categories:')
print(list(newsgroups.target_names))
print("\n")

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

from google.colab import drive
drive.mount('/content/gdrive')

with open('/content/gdrive/My Drive/437-A4/SMSSpamCollection.txt', 'r') as f: 
    # print(f.read())
    df = pd.read_csv(f, sep = "\t", header=None, names=["mtypes", "ms"])
categories = ['mtypes', 'ms']
X = df.loc[:, categories]
print(X.shape)
newsgroups = df(categories=categories)
print('List of newsgroup categories:')
print(list(newsgroups.target_names))
print("\n")