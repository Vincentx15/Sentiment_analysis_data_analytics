import time as t
from classifier import *
from features import load_features

# Parameters
classifier = 'LSTM'
method = 'we'
language = 'fr'
epochs = 5
batch_size = 32
duration = 3600 * 6

# Initialization
t0 = t.time()
best_model, best_mse = None, 0
save_fname = 'data/model/test_3_random_trained_' + classifier + '_' + method + '_' + language

# Load the data
x_train, x_test, y_train, y_test = load_features(language, method)
validation_data = (x_test, y_test)

# Loop for the specified duration
cmpt = 1
while t.time() - t0 < duration:

    print("Trial number {}:".format(cmpt))

    # Create a random classifier
    new_model, new_info = create_random_classifier(classifier)

    # Train the classifier
    print("Training...")
    new_model, history = train_classifier(classifier, new_model, x_train, y_train, epochs, batch_size,
                                          validation_data=validation_data, save_file=None, return_history=True,
                                          callback=False, verbose=0)
    val_mse = history.history['val_mean_squared_error']
    new_mse, new_epochs = min(val_mse), np.argmin(val_mse) + 1

    # Update the saved model
    if (not best_model) or (new_mse < best_mse):
        best_model = new_model
        best_info = new_info
        best_mse = new_mse
        best_epochs = new_epochs

        save_classifier(classifier, best_model, save_fname)
        print("Best model information: {}".format(best_info))
        print("MSE: {}".format(best_mse))
        print("Epochs: {}".format(best_epochs))

    else:
        print("Training done, MSE: {} (best MSE: {})".format(new_mse, best_mse))

    cmpt += 1
