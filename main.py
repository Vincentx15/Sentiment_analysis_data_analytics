from classifier import *
from features import load_features

# Parameters
classifier = 'LSTM'
method = 'we'
language = 'en'

# Load the data
# x_train, x_test, y_train, y_test = load_features(language, method)
# x_test = np.load('data/features/twitter_fr_all.npy')
# x_test = np.load('data/features/en.npy')
x_test = np.load('data/features/twitter_en_yellowvest.npy')

# Load a classifier
# load_file = 'data/model/untrained_' + classifier
# load_file = 'data/model/trained_' + classifier + '_' + method + '_' + language
load_file = 'data/model/trained_' + classifier + '_' + method + '_' + language

model = load_classifier(classifier, load_file)

'''
# Train the classifier
epochs = 14
batch_size = 16
validation_data = (x_test, y_test)
save_file = 'data/model/test_3_retrained_bis_random_trained_' + classifier + '_' + method + '_' + language
model = train_classifier(classifier, model, x_train, y_train, epochs, batch_size, validation_data,
                         save_file=save_file+'.h5')

# Predict on the test set
model = load_classifier(classifier, save_file)
'''

y_pred = predict_classifier(model, x_test)
np.save('data/twitter/results/old_yellowvest_'+classifier+'_'+method+'_'+language, y_pred)

'''
# Compute the performance
print("RMSE: {}".format(evaluate_classifier("RMSE", y_test, y_pred)))
print("Accuracy: {}".format(evaluate_classifier("multi_accuracy", y_test, y_pred)))
print("Binary accuracy: {}".format(evaluate_classifier("binary_accuracy", y_test, y_pred)))
'''
