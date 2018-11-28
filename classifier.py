import numpy as np
import math
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping


def create_classifier(classifier_type):
    """
    Return a compiled classifier with the parameters specified inside the function
    :param classifier_type: string, define which classifier to use
    :return: untrained classifier
    """

    if classifier_type == "SVM":

        # Parameters
        gamma = 'scale'
        verbose = True

        # Create classifier
        m = svm.SVR(gamma=gamma, verbose=verbose)

        print("Model created.")
        return m

    elif classifier_type == "NN":

        # Parameters
        input_dim = 1573                        # Int
        layers = [256, 64, 1]                   # List of int
        activations = ['relu', 'relu', 'relu']  # List of strings
        dropout = 0.5                           # Float
        loss = 'mean_squared_error'             # String
        optimizer = 'sgd'                       # String

        # Create model
        m = Sequential()
        m.add(Dense(units=layers[0], activation=activations[0], input_dim=input_dim))
        for i in range(1, len(layers)):
            if dropout != 0:
                m.add(Dropout(dropout))
            m.add(Dense(units=layers[i], activation=activations[i]))
        m.compile(loss=loss, optimizer=optimizer)

        print("Model created.")
        return m

    elif classifier_type == "LSTM":

        # Parameters
        input_shape = (42, 300)             # Tuple of int
        cells = 2                           # Int
        units = [32, 16]                   # List of int
        return_sequences = [True, False]    # List of bool
        activation = 'tanh'                 # String
        dropout = 0.5                       # Float
        loss = 'mean_squared_error'         # String
        optimizer = 'adam'                  # String
        nn_layers = [8, 1]                 # List of int
        nn_activations = ['relu', 'relu']   # List of string

        # Create model
        m = Sequential()
        m.add(GRU(input_shape=input_shape, units=units[0], activation=activation, dropout=dropout,
                  return_sequences=return_sequences[0]))
        for i in range(1, cells):
            m.add(GRU(units=units[i], activation=activation, dropout=dropout, return_sequences=return_sequences[i]))
        for i in range(len(nn_layers)):
            m.add(Dense(units=nn_layers[i], activation=nn_activations[i]))
        m.compile(loss=loss, optimizer=optimizer)

        print("Model created.")
        return m

    else:
        raise ValueError("Wrong classifier_type defined.")


def train_classifier(classifier_type, m, x, y, ep=None, b_s=None, validation_data=None, save_file=None):
    """
    Train the model m on the values of x_test and y_train
    :param m: model to train
    :param x: array, train set
    :param y: array, label set
    :param ep: int, epochs
    :param b_s: int, size of the batch
    :param validation_data: tuple of array, validation set
    :param save_file: string, where to save the file
    :return: trained model
    """
    if classifier_type == "SVM":
        m.fit(x, y)
        print("Model trained.")
        return

    elif classifier_type in ["NN", "LSTM"]:
        checkpoint = ModelCheckpoint(save_file, save_best_only=True)
        stopping = EarlyStopping(min_delta=0.1, patience=5)
        m.fit(x=x, y=y, epochs=ep, batch_size=b_s, validation_data=validation_data, callbacks=[checkpoint, stopping])
        print("Model trained.")
        return m

    else:
        raise ValueError("Wrong classifier.")


def predict_classifier(m, x):
    """
    Compute the prediction of the classifier in the specified data
    :param m: model that performs the prediction
    :param x: array, data used for the prediction
    :return: array, prediction
    """
    pred = m.predict(x)
    print("Prediction done.")
    return pred


def evaluate_classifier(measure_type, y_true, y_pred):
    """
    Evaluate the model with the specified measure_type
    :param m: model to evaluate
    :param measure_type: string, measure to use
    :param x: array, data for the prediction
    :param y: true data
    :return: corresponding score or error
    """
    if measure_type == "RMSE":
        return math.sqrt(mean_squared_error(y_true, y_pred))

    elif measure_type == "binary_accuracy":
        return accuracy_score(np.round(np.divide(y_true, 5.)), np.round(np.divide(y_pred, 5.)))

    elif measure_type == "multi_accuracy":
        return accuracy_score(np.round(y_true), np.round(y_pred))

    else:
        raise ValueError("Wrong classifier defined.")


def save_classifier(classifier_type, m, fname):
    """
    Function that saves the model
    :param classifier_type: string, type of classifier
    :param m: model to save
    :param fname: string, name of the file without the extension
    :return: /
    """
    if classifier_type == "SVM":
        dump(m, fname + '.joblib')
        print("Model saved.")

    elif classifier_type in ["NN", "LSTM"]:
        m.save(fname + '.h5')
        print("Model saved.")

    else:
        raise ValueError("Wrong classifier.")


def load_classifier(classifier_type, fname):
    """
    Function that loads and returns a model
    :param classifier_type: string, type of classifier
    :param fname: string, name of the file without the extension
    :return: model saved
    """
    if classifier_type == "SVM":
        print("Model loaded.")
        return load(fname + '.joblib')

    elif classifier_type in ["NN", "LSTM"]:
        print("Model loaded.")
        return load_model(fname + '.h5')

    else:
        raise ValueError("Wrong classifier_type defined.")


if __name__ == '__main__':

    # Parameters
    classifier = 'LSTM'  # String, can be SVM, NN or LSTM

    # Create a classifier
    model = create_classifier(classifier)

    # Save model
    save_file = 'data/model/untrained_' + classifier  # String, name of the file to save/load with no extension
    save_classifier(classifier, model, save_file)
