from classifier import *
from features import load_features

# Parameters
classifier = 'LSTM'
method = 'we'
language = 'fr'

# Load the data
x_train, x_test, y_train, y_test = load_features(language, method)

# Load a classifier
load_file = 'data/model/untrained_' + classifier
# load_file = 'data/model/trained_' + classifier + '_' + method + '_' + language
model = load_classifier(classifier, load_file)

# Train the classifier
epochs = 20
batch_size = 32
validation_data = (x_test, y_test)
save_file = 'data/model/trained_' + classifier + '_' + method + '_' + language
model = train_classifier(classifier, model, x_train, y_train, epochs, batch_size, validation_data,
                         save_file=save_file+'.h5')

# Predict on the test set
model = load_classifier(classifier, save_file)
y_pred = predict_classifier(model, x_test)

# Compute the performance
print("RMSE: {}".format(evaluate_classifier("RMSE", y_test, y_pred)))
print("Accuracy: {}".format(evaluate_classifier("multi_accuracy", y_test, y_pred)))
print("Binary accuracy: {}".format(evaluate_classifier("binary_accuracy", y_test, y_pred)))

# # Save the model
# save_classifier(classifier, model, save_file)
