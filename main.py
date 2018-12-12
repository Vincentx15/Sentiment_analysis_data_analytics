from classifier import *
from features import load_features, save_prediction

""" Parameters """
classifier = 'LSTM'
method = 'we'
database = 'allocine'
language = 'fr'
query = ''
extended = False

""" Load the data """
x_train, x_test, y_train, y_test = load_features(database, language, method)
# x_test = load_features(database, language, method, query, extended)
x_test = np.concatenate((x_train, x_test), axis=0)

""" Load a classifier """
# load_file = 'data/model/untrained_' + classifier
load_file = 'data/model/best_trained_' + classifier + '_' + method + '_' + language
model = load_classifier(classifier, load_file)

'''
""" Train the classifier """
epochs = 5
batch_size = 16
validation_data = (x_test, y_test)
save_file = 'data/model/trained_' + classifier + '_' + method + '_' + language
model = train_classifier(classifier, model, x_train, y_train, epochs, batch_size, validation_data,
                         save_file=save_file+'.h5')
model = load_classifier(classifier, save_file)
'''

""" Make a prediction """
y_pred = predict_classifier(model, x_test)
save_prediction(y_pred, database, classifier, method, language, query, extended)

'''
# Compute the performance
print("RMSE: {}".format(evaluate_classifier("RMSE", y_test, y_pred)))
print("Accuracy: {}".format(evaluate_classifier("multi_accuracy", y_test, y_pred)))
print("Binary accuracy: {}".format(evaluate_classifier("binary_accuracy", y_test, y_pred)))
'''
