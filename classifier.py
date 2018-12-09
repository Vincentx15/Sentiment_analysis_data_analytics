import numpy as np
import random as rd
import math
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping


def train_classifier(classifier_type, m, x, y, ep=None, b_s=None, validation_data=None, save_file=None,
                     return_history=False, callback=True, verbose=1):
    """
    Train the model m on the values of x_test and y_train
    :param m: model to train
    :param x: array, train set
    :param y: array, label set
    :param ep: int, epochs
    :param b_s: int, size of the batch
    :param validation_data: tuple of array, validation set
    :param save_file: string, where to save the file
    :param return_history: if True, return the history
    :return: trained model
    """
    if classifier_type == "SVM":
        m.fit(x, y)
        print("Model trained.")
        return

    elif classifier_type in ["NN", "LSTM"]:
        if callback:
            checkpoint = ModelCheckpoint(save_file, save_best_only=True)
            stopping = EarlyStopping(min_delta=0.1, patience=3)
            history = m.fit(x=x, y=y, epochs=ep, batch_size=b_s, validation_data=validation_data,
                            callbacks=[checkpoint, stopping], verbose=verbose)
        else:
            history = m.fit(x=x, y=y, epochs=ep, batch_size=b_s, validation_data=validation_data, verbose=verbose)
        print("Model trained.")
        if return_history:
            return m, history
        else:
            return m

    else:
        raise ValueError("Wrong classifier.")


def create_random_classifier(classifier_type):
    """
    Return a compiled classifier with random parameters
    :param classifier_type: string, define which classifier to use
    :return: untrained classifier
    """

    if classifier_type == "NN":

        # Fixed parameters
        input_dim = 1573

        # Random parameters
        possible_layers_nb = range(1, 5)
        possible_layers = [2**k for k in range(1, 11)]
        possible_activations = ['relu', 'tanh', 'softmax', 'elu', 'sigmoid', 'linear']
        possible_dropouts = np.arange(0.0, 0.6, 0.1)
        possible_optimizer = ['sgd', 'Adagrad', 'Adadelta', 'Adam']

        layers_nb = rd.choice(possible_layers_nb)
        layers = rd.sample(possible_layers, layers_nb-1)
        layers.sort(reverse=True)
        activations = [rd.choice(possible_activations) for _ in range(layers_nb-1)]
        dropout = rd.choice(possible_dropouts)
        optimizer = rd.choice(possible_optimizer)

        # Define the info
        info = {
            'layers_nb': layers_nb,
            'layers': layers,
            'activations': activations,
            'dropout': dropout,
            'optimizer': optimizer
        }

        # Create model
        m = Sequential()
        m.add(Dense(units=layers[0], activation=activations[0], input_dim=input_dim))
        m.add(Dropout(dropout))
        for i in range(1, layers_nb-1):
            m.add(Dense(units=layers[i], activation=activations[i]))
            m.add(Dropout(dropout))
        m.add(Dense(units=1, activation='relu'))
        m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

        print("Random model created.")
        return m, info

    elif classifier_type == "LSTM":

        # Fixed parameters
        input_shape = (42, 300)

        # Random parameters
        possible_cells_nb = range(1, 5)
        possible_units = [2**k for k in range(0, 7)]
        possible_activations = ['relu', 'tanh', 'softmax', 'elu', 'sigmoid', 'linear']
        possible_dropouts = np.arange(0.0, 0.6, 0.1)
        possible_optimizer = ['RMSprop', 'sgd', 'Adagrad', 'Adadelta', 'Adam']

        cells = rd.choice(possible_cells_nb)
        units = rd.sample(possible_units, cells)
        units.sort(reverse=True)
        return_sequences = [True for _ in range(cells-1)] + [False]
        activations = [rd.choice(possible_activations) for _ in range(cells)]
        dropout = rd.choice(possible_dropouts)
        optimizer = rd.choice(possible_optimizer)

        nn_input_size = units[-1]
        possible_nn_layers_nb = [k for k in range(1, 4) if (2**(k-1) <= nn_input_size)]
        possible_nn_layers = [2**k for k in range(0, 7) if (2**k <= nn_input_size)]

        if nn_input_size == 1:
            nn_layers_nb = 0
            nn_layers = []
            nn_activations = []
        else:
            nn_layers_nb = rd.choice(possible_nn_layers_nb)
            nn_layers = rd.sample(possible_nn_layers, nn_layers_nb-1)+[1]
            nn_layers.sort(reverse=True)
            nn_activations = [rd.choice(possible_activations) for _ in range(nn_layers_nb-1)]+['relu']

        # Define the info
        info = {
            'cells': cells,
            'units': units,
            'return_sequences': return_sequences,
            'activations': activations,
            'dropouts': dropout,
            'optimizer': optimizer,
            'nn_layers_nb': nn_layers_nb,
            'nn_layers': nn_layers,
            'nn_activations': nn_activations
        }

        # Create model
        m = Sequential()
        m.add(GRU(input_shape=input_shape, units=units[0], activation=activations[0], dropout=dropout,
                  return_sequences=return_sequences[0]))
        for i in range(1, cells):
            m.add(GRU(units=units[i], activation=activations[i], dropout=dropout, return_sequences=return_sequences[i]))
        for i in range(nn_layers_nb):
            if i != 0:
                m.add(Dropout(dropout))
            m.add(Dense(units=nn_layers[i], activation=nn_activations[i]))
        m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

        print("Model created.")
        return m, info
    else:
        raise ValueError("Wrong classifier_type defined.")


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


'''
if __name__ == '__main__':

    # Parameters
    classifier = 'LSTM'  # String, can be SVM, NN or LSTM

    # Create a classifier
    model = create_classifier(classifier)

    # Save model
    save_file = 'data/model/untrained_' + classifier  # String, name of the file to save/load with no extension
    save_classifier(classifier, model, save_file)
'''
