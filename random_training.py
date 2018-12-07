import time as t
from classifier import *
from features import load_features

# Parameters
classifier = 'LSTM'
method = 'we'
language = 'fr'
duration = 3600*2
epochs = 10
batch_size = 32

# Initialization
t0 = t.time()
best_model = None

# Load the data
x_train, x_test, y_train, y_test = load_features(language, method)
validation_data = (x_test, y_test)

# Loop for the specified duration
while (t.time-t0) <= duration:

    # Create a random classifier
    new_model, new_info = create_random_classifier(classifier)

    # Train the classifier
    new_model, history = train_classifier(classifier, new_model, x_train, y_train, epochs, batch_size, validation_data,
                                          True)
    val_losses = history.history['val_loss']
    new_mse, new_epochs = min(val_losses), np.argmin(val_losses) + 1

    # Update the saved model
    if (not best_model) or (new_mse < best_mse):
        best_model = new_model
        best_info = new_info
        best_mse = new_mse
        best_epochs = new_epochs


print("Best model information: {}".format(best_info))
