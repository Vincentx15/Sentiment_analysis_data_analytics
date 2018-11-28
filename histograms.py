from classifier import *
from features import load_features

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from textblob import TextBlob

language = 'en'

# Compute the true values
_, _, y_train, y_test = load_features(language, 'bow')
y_true = np.concatenate((y_train, y_test), axis=0)

classifiers = ['SVM', 'NN', 'LSTM']
methods = ['bow', 'bow', 'we']
y_pred = []

# for i in range(3):
#     print('')
#     print("{}/3...".format(i+1))
#     method = methods[i]
#     classifier = classifiers[i]
#     x_train, x_test, _, _ = load_features(language, method)
#     load_file = 'data/model/trained_' + classifier + '_' + method + '_' + language
#     model = load_classifier(classifier, load_file)
#     y_pred_train = predict_classifier(model, x_train)
#     y_pred_test = predict_classifier(model, x_test)
#     y_pred.append(np.concatenate((y_pred_train, y_pred_test), axis=0))

print("Textblob...")

data_en = pd.read_csv('data/raw_csv/imdb.csv')
reviews_en = data_en['review']
ratings_en = data_en['rating']
results_en = [TextBlob(review).sentiment[0] for review in reviews_en]
results_en = [2.5 * (res + 1) for res in results_en]
results_en = np.array(results_en)

sns.distplot(y_true, hist=False, kde_kws={"color": "b", "lw": 2, "label": "True"})
for i in range(1, 3):
    y_pred = np.load('data/saved_distributions/distributions_'+language+'_'+str(i)+'.npy')
    sns.distplot(y_pred, hist=False, kde_kws={"label": classifiers[i]})

sns.distplot(results_en, hist=False, kde_kws={"label": 'Textblob en'})

# data_fr = pd.read_csv('data/raw_csv/allocine.csv')
# reviews_fr = data_fr['review']
# ratings_fr = data_fr['rating']
# results_fr = [TextBlob(review).sentiment[0] for review in reviews_fr]
# results_fr = [2.5 * (res + 1) for res in results_fr]
# results_fr = np.array(results_fr)
# sns.distplot(results_fr, hist=False, kde_kws={"label": 'Textblob Fr'})
#
#
#

plt.show()

