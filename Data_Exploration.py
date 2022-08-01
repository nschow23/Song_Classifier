import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

def adversarial_validation(X, Y, n_splits=10):
    # Combine both datasets
    sparse_merge = sparse.vstack((X, Y))

    # Label the datasets
    y = np.array([0 for _ in range(X.shape[0])] + [1 for _ in range(Y.shape[0])])

    # Do 10 Fold CV
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    lr_auc = np.array([])


    for train_idx, test_idx in kfold.split(sparse_merge, y):
        # Run Log Reg
        x_train, y_train = sparse_merge[train_idx], y[train_idx]
        x_test, y_test = sparse_merge[test_idx], y[test_idx]

        log_reg = SGDClassifier(loss='log')
        log_reg.fit(x_train, y_train)
        y_test_prob = log_reg.predict_proba(x_test)[:, 1]
        lr_auc = np.append(lr_auc, roc_auc_score(y_test, y_test_prob))

    print('Logistic Regression AUC: {:.3f}'.format(lr_auc.mean()))

### load data and create test and train dataset
df = pd.read_csv("lyrics_plus.csv", header=None)
df.columns = ['artist', 'lyrics', 'label']
# print(df.head())
# print(df.shape)
lyrics = df.lyrics
labels = df.label
# print(lyrics.head())
# print(labels.head())

pop_lyrics = df.lyrics[df.label ==0]
pop_label = df.label[df.label==0]
country_lyrics = df.lyrics[df.label == 1]
country_label = df.label[df.label == 1]

## create train and test set
train, test, y_train, y_test = train_test_split(lyrics, labels, test_size=0.3)
#
# #vectorize data
# bow = CountVectorizer()
# x_train = bow.fit_transform(train.values)
# x_test = bow.transform(test.values)
#
# #### implement adversarial validation
# adversarial_validation(x_train, x_test[:50])
#
# ####save train and test data set
train_set = np.column_stack((train, y_train))
test_set = np.column_stack((test, y_test))
#
#
np.savetxt('lyrics_train', train_set, fmt=('%s'), delimiter=',')
np.savetxt('lyrics_test', test_set, fmt=('%s'), delimiter=',')






