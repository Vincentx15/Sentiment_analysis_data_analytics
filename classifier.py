# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:28:23 2018

@author: Clément Jumel
"""

import numpy as np
import math as m
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense


def train_classifier(classifier, x_train, y_train, parameters = None):
    """
    Return a classifier with the specified parameters trained on x_train and y_train
    :param classifier: string, define which classifier to use
    :param x_train: array, train set
    :param y_train: array, label set
    :param parameters: dictionnary, define which parameter to use
    :return: trained classifier
    """

    if classifier == "SVM":
        model = svm.SVR()
        model.fit(x_train, y_train)
        return model

    elif classifier == "NN":
        try:
            input_dim = parameters['NN_input_dim']
            layers = parameters['NN_layers']
            activations = parameters['NN_activations']
            loss = parameters['NN_loss']
            optimizer = parameters['NN_optimizer']
            epochs = parameters['epochs']
            batch_size = parameters['batch_size']
        except TypeError:
            raise ValueError("Parameters not defined")

        model = Sequential()
        model.add(Dense(units=layers[0], activation=activations[0], input_dim=input_dim))
        for i in range(1, len(layers)):
            model.add(Dense(units=layers[i], activation=activations[i]))

        model.compile(loss=loss, optimizer=optimizer)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return model

    else:
        raise ValueError("Wrong classifier defined.")


def predict_classifier(model, x_pred):
    """
    Compute the prediction of the classifier in the specified data
    :param model: model that performs the prediction
    :param x_pred: array, data used for the prediction
    :return: array, prediction
    """
    return model.predict(x_pred)


def evaluate_classifier(model, measure, x_test, y_test):
    """
    Evaluate the model with the specified measure
    :param model: model to evalute
    :param measure: string, measure to use
    :param x_test: array, data for the prediction
    :param y_test: true data
    :return: score or error
    """
    if measure == "RMSE":
        y_pred = predict_classifier(model, x_test)
        return rmse(y_pred, y_test)

    else:
        raise ValueError("Wrong classifier defined.")


def rmse(y_pred, y_test):
    """
    Root mean square error
    :param y_pred: array, first data to compare
    :param y_test: array, second data to compare
    :return: float, RMSE of the data
    """
    return m.sqrt(np.mean(np.square(y_pred - y_test)))


if __name__ == '__main__':

    Classifier = "NN"
    Measure = "RMSE"

    Parameters = {'NN_input_dim': 10,
                  'NN_layers': [8, 4, 1],
                  'NN_activations': ['relu', 'relu', 'relu'],
                  'NN_loss': 'mean_squared_error',
                  'NN_optimizer': 'sgd',
                  'epochs': 10,
                  'batch_size': 1}

    X_train = np.asarray([[0, 0, 1, 2, 3, 1, 4, 4, 3, 4],
                          [0, 0, 1, 2, 3, 1, 4, 4, 3, 4]])
    Y_train = [0.5, 2.5]
    # X_test = [[0.6, -0.1], [1.8, 2.2]]
    # Y_test = [0.3, 2.8]
    # X_pred = [[0., 1.]]

    # Train a classifier
    Model = train_classifier(Classifier, X_train, Y_train, Parameters)

    # # Evaluate the classifier
    # error = evaluate_classifier(Model, Measure, X_test, Y_test)
    #
    # # Predict a value
    # Y_pred = predict_classifier(Model, X_pred)
    # print("Error: {}".format(error))
