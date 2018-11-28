from classifier import *
from features import load_features

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


language = 'fr'

# Compute the true values
_, _, y_train, y_test = load_features(language, 'bow')
y_true = np.concatenate((y_train, y_test), axis=0)

classifiers = ['SVM', 'NN', 'LSTM']
methods = ['bow', 'bow', 'we']
y_pred = []

for i in range(3):
    print('')
    print("{}/3...".format(i+1))
    method = methods[i]
    classifier = classifiers[i]
    x_train, x_test, _, _ = load_features(language, method)
    load_file = 'data/model/trained_' + classifier + '_' + method + '_' + language
    model = load_classifier(classifier, load_file)
    y_pred_train = predict_classifier(model, x_train)
    y_pred_test = predict_classifier(model, x_test)
    y_pred.append(np.concatenate((y_pred_train, y_pred_test), axis=0))

sns.distplot(y_true, hist=False, kde_kws={"color": "r", "lw": 3, "label": "True"})
for i in range(1, 3):
    y_pred = np.load('data/saved_distributions/distributions_'+language+'_'+str(i)+'.npy')
    sns.distplot(y_pred, hist=False, kde_kws={"label": classifiers[i]})
plt.show()

