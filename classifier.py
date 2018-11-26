import numpy as np
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM


# Default parameters
classifier = "LSTM"                 # String, can be SVM, NN or LSTM
measure = "MSE"                     # String, can be RMSE or accuracy
file_name = "data/model/model1"     # String, name of the file to save/load with no extension
epochs = 1                          # Epochs for the training
batch_size = 1                      # Size of the batch for the training

parameters = {
    'SVM': None,
    'NN': {'input_dim': 10,                             # Int
           'layers': [8, 4, 1],                         # List of int
           'activations': ['relu', 'relu', 'relu'],     # List of strings
           'dropout': 0.5,                              # Float
           'loss': 'mean_squared_error',                # String
           'optimizer': 'sgd',                          # String
           },
    'LSTM': {'input_shape': (10, 3),                    # Tuple of int
             'cells': 2,                                # Int
             'units': [2,1],                            # List of int
             'return_sequences': [True, False],         # List of bool
             'activation': 'tanh',                      # String
             'dropout': 0.2,                            # Float
             'loss': 'mean_squared_error',              # String
             'optimizer': 'adam',                       # String
             'metrics': ['mean_squared_error'],         # List of string
             }
}


def create_classifier(classifier_type, classifier_param=None):
    """
    Return a compiled classifier with the specified parameters
    :param classifier_type: string, define which classifier to use
    :param classifier_param: dictionary, define which parameter to use
    :return: trained classifier
    """

    if classifier_type == "SVM":
        model = svm.SVR()
        return model

    elif classifier_type == "NN":
        try:
            input_dim = classifier_param[classifier_type]['input_dim']
            layers = classifier_param[classifier_type]['layers']
            activations = classifier_param[classifier_type]['activations']
            dropout = classifier_param[classifier_type]['dropout']
            loss = classifier_param[classifier_type]['loss']
            optimizer = classifier_param[classifier_type]['optimizer']
        except TypeError:
            raise ValueError("parameters not defined")

        model = Sequential()
        model.add(Dense(units=layers[0], activation=activations[0], input_dim=input_dim))
        for i in range(1, len(layers)):
            if dropout != 0:
                model.add(Dropout(dropout))
            model.add(Dense(units=layers[i], activation=activations[i]))

        model.compile(loss=loss, optimizer=optimizer)
        return model

    elif classifier_type == "LSTM":
        try:
            input_shape = classifier_param[classifier_type]['input_shape']
            cells = classifier_param[classifier_type]['cells']
            units = classifier_param[classifier_type]['units']
            return_sequences = classifier_param[classifier_type]['return_sequences']
            activation = classifier_param[classifier_type]['activation']
            dropout = classifier_param[classifier_type]['dropout']
            loss = classifier_param[classifier_type]['loss']
            optimizer = classifier_param[classifier_type]['optimizer']
            metrics = classifier_param[classifier_type]['metrics']
        except TypeError:
            raise ValueError("parameters not defined")

        model = Sequential()
        model.add(LSTM(input_shape=input_shape, units=units[0], activation=activation, dropout=dropout, return_sequences=return_sequences[0]))
        for i in range(1, cells):
            model.add(LSTM(units=units[i], activation=activation, dropout=dropout, return_sequences=return_sequences[i]))

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    else:
        raise ValueError("Wrong classifier_type defined.")


def train_classifier(m, x, y, ep=None, b_s=None):
    """
    Train the model m on the values of x_test and y_train
    :param m: model to train
    :param x: array, train set
    :param y: array, label set
    :param ep: int, epochs
    :param b_s: int, size of the batch
    :return:
    """
    if (ep is not None) and (b_s is not None):
        m.fit(x, y, epochs=ep, batch_size=b_s)
    else:
        m.fit(x, y)
    return m


def predict_classifier(m, x):
    """
    Compute the prediction of the classifier in the specified data
    :param m: model that performs the prediction
    :param x: array, data used for the prediction
    :return: array, prediction
    """
    return m.predict(x)


def evaluate_classifier(m, measure_type, x, y):
    """
    Evaluate the model with the specified measure_type
    :param m: model to evaluate
    :param measure_type: string, measure to use
    :param x: array, data for the prediction
    :param y: true data
    :return: corresponding score or error
    """
    if measure_type == "MSE":
        y_pred = predict_classifier(m, x)
        return mean_squared_error(y, y_pred)

    if measure_type == "accuracy":
        y_pred = predict_classifier(m, x)
        return accuracy_score(y, y_pred)

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

    elif classifier_type in ["NN", "LSTM"]:
        m.save(fname + '.h5')

    else:
        raise ValueError("Wrong classifier_type defined.")


def load_classifier(classifier_type, fname):
    """
    Function that loads and returns a model
    :param classifier_type: string, type of classifier
    :param fname: string, name of the file without the extension
    :return: model saved
    """
    if classifier_type == "SVM":
        return load(fname + '.joblib')

    elif classifier_type in ["NN", "LSTM"]:
        return load_model(fname + '.h5')

    else:
        raise ValueError("Wrong classifier_type defined.")


if __name__ == '__main__':

    # Initialize random data
    x_train = np.random.rand(5, 10, 3)
    y_train = np.random.rand(5, 1)

    x_test = np.random.rand(3, 10, 3)
    y_test = np.random.rand(3, 1)

    x_pred = np.random.rand(3, 10, 3)

    # Create a classifier
    print("Create the classifier...")
    model = create_classifier(classifier, parameters)

    # Train the classifier
    print("Train the classifier")
    model = train_classifier(model, x_train, y_train, epochs, batch_size)

    # Evaluate the classifier
    print("Evaluate the classifier")
    error = evaluate_classifier(model, measure, x_test, y_test)

    # Predict a value
    print("Predict a value")
    y_pred = predict_classifier(model, x_pred)
    print("Error: {}".format(error))

    # Save model
    print("Save the model...")
    save_classifier(classifier, model, file_name)

    # Load model
    print("Load the model...")
    model = load_classifier(classifier, file_name)
