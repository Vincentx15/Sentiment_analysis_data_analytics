from classifier import *
from features import load_features
import seaborn as sns
import numpy as np

# Parameters
classifier = 'LSTM'
method = 'we'
language = 'en'

# Load the file
load_file = 'data/model/trained_' + classifier + '_' + method + '_' + language
model = load_classifier(classifier, load_file)

# Load the data
x_train, x_test, y_train, y_test = load_features(language, method)

# Predict on the sets
y_pred_train = predict_classifier(model, x_train)
y_pred_test = predict_classifier(model, x_test)

# Compute the performance
print("RMSE (train): {}".format(evaluate_classifier("RMSE", y_train, y_pred_train)))
print("Accuracy (train): {}".format(evaluate_classifier("multi_accuracy", y_train, y_pred_train)))
print("Binary accuracy (train): {}".format(evaluate_classifier("binary_accuracy", y_train, y_pred_train)))
print("RMSE (test): {}".format(evaluate_classifier("RMSE", y_test, y_pred_test)))
print("Accuracy (test): {}".format(evaluate_classifier("multi_accuracy", y_test, y_pred_test)))
print("Binary accuracy (test): {}".format(evaluate_classifier("binary_accuracy", y_test, y_pred_test)))

# Compute the histograms
import matplotlib.pyplot as plt
y_pred = np.concatenate((y_pred_train, y_pred_test), axis=0)
y_true = np.concatenate((y_train, y_test), axis=0)
sns.distplot(y_pred)
plt.show()

sns.distplot(y_true)
plt.show()