# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:28:23 2018

@author: Clément Jumel
"""

import numpy as np
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM


# Default parameters
Classifier = "LSTM"             # String, can be SVM, NN or LSTM
Measure = "MSE"                 # String, can be RMSE or accuracy
Fname = "data/model/model1"     # String, name of the file to save/load with no extension

Parameters = {
    'SVM': None,
    'NN': {'input_dim': 10,                             # Int
           'layers': [8, 4, 1],                         # List of int
           'activations': ['relu', 'relu', 'relu'],     # List of strings
           'dropout': 0.5,                              # Float
           'loss': 'mean_squared_error',                # String
           'optimizer': 'sgd',                          # String
           'epochs': 10,                                # Int
           'batch_size': 1                              # Int
           },
    'LSTM': {'input_shape': (10, 3),
             'cells': 2,
             'units': [2,1],
             'return_sequences': [True, False],
             'activation': 'tanh',
             'dropout': 0.2,
             'loss': 'mean_squared_error',
             'optimizer': 'adam',
             'metrics': ['mean_squared_error'],
             'epochs': 3,
             'batch_size': 1
             }
}


def train_classifier(classifier, x_train, y_train, parameters=None):
    """
    Return a classifier with the specified parameters trained on x_train and y_train
    :param classifier: string, define which classifier to use
    :param x_train: array, train set
    :param y_train: array, label set
    :param parameters: dictionary, define which parameter to use
    :return: trained classifier
    """

    if classifier == "SVM":
        model = svm.SVR()
        model.fit(x_train, y_train)
        return model

    elif classifier == "NN":
        try:
            input_dim = parameters[classifier]['input_dim']
            layers = parameters[classifier]['layers']
            activations = parameters[classifier]['activations']
            dropout = parameters[classifier]['dropout']
            loss = parameters[classifier]['loss']
            optimizer = parameters[classifier]['optimizer']
            epochs = parameters[classifier]['epochs']
            batch_size = parameters[classifier]['batch_size']
        except TypeError:
            raise ValueError("Parameters not defined")

        model = Sequential()
        model.add(Dense(units=layers[0], activation=activations[0], input_dim=input_dim))
        for i in range(1, len(layers)):
            if dropout != 0:
                model.add(Dropout(dropout))
            model.add(Dense(units=layers[i], activation=activations[i]))

        model.compile(loss=loss, optimizer=optimizer)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return model

    elif classifier == "LSTM":
        try:
            input_shape = parameters[classifier]['input_shape']
            cells = parameters[classifier]['cells']
            units = parameters[classifier]['units']
            return_sequences = parameters[classifier]['return_sequences']
            activation = parameters[classifier]['activation']
            dropout = parameters[classifier]['dropout']
            loss = parameters[classifier]['loss']
            optimizer = parameters[classifier]['optimizer']
            metrics = parameters[classifier]['metrics']
            epochs = parameters[classifier]['epochs']
            batch_size = parameters[classifier]['batch_size']
        except TypeError:
            raise ValueError("Parameters not defined")

        model = Sequential()
        model.add(LSTM(input_shape=input_shape, units=units[0], activation=activation, dropout=dropout, return_sequences=return_sequences[0]))
        for i in range(1, cells):
            model.add(LSTM(units=units[i], activation=activation, dropout=dropout, return_sequences=return_sequences[i]))

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
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
    if measure == "MSE":
        y_pred = predict_classifier(model, x_test)
        return mean_squared_error(y_test, y_pred)

    if measure == "accuracy":
        y_pred = predict_classifier(model, x_test)
        return accuracy_score(y_test, y_pred)

    else:
        raise ValueError("Wrong classifier defined.")


def save_classifier(classifier, model, fname):
    """
    Function that saves the model
    :param classifier: string, type of classifier
    :param model: model to save
    :param fname: string, name of the file without the extension
    :return: /
    """
    if classifier == "SVM":
        dump(model, fname + '.joblib')

    elif classifier in ["NN", "LSTM"]:
        model.save(fname + '.h5')

    else:
        raise ValueError("Wrong classifier defined.")


def load_classifier(classifier, fname):
    """
    Function that loads and returns a model
    :param classifier: string, type of classifier
    :param fname: string, name of the file without the extension
    :return: model saved
    """
    if classifier == "SVM":
        return load(fname + '.joblib')

    elif classifier in ["NN", "LSTM"]:
        return load_model(fname + '.h5')

    else:
        raise ValueError("Wrong classifier defined.")


if __name__ == '__main__':

    import random as rd
    # Data
    print("Specify training set...")

    X_train = np.random.rand(5, 10, 3)
    Y_train = np.random.rand(5, 1)

    X_test = np.random.rand(3, 10, 3)
    Y_test = np.random.rand(3, 1)
    X_pred = np.random.rand(3, 10, 3)

    # Train a classifier
    print("Train the classifier...")
    Model = train_classifier(Classifier, X_train, Y_train, Parameters)

    # Evaluate the classifier
    print("Evaluate the classifier")
    error = evaluate_classifier(Model, Measure, X_test, Y_test)

    # Predict a value
    print("Predict a value")
    Y_pred = predict_classifier(Model, X_pred)
    print("Error: {}".format(error))

    # Save model
    print("Save the model...")
    save_classifier(Classifier, Model, Fname)

    # Load model
    print("Load the model...")
    Model = load_classifier(Classifier, Fname)
