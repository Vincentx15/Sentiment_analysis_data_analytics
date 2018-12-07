import time as t
from classifier import *
from features import load_features

# Parameters
classifier = 'LSTM'
method = 'we'
language = 'fr'
duration = 3600*1
epochs = 20
batch_size = 32

# Initialization
t0 = t.time()
best_model = None

# Load the data
x_train, x_test, y_train, y_test = load_features(language, method)
validation_data = (x_test, y_test)

# Loop for the specified duration
while (t.time()-t0) <= duration:

    # Create a random classifier
    new_model, new_info = create_random_classifier(classifier)

    # Train the classifier
    new_model, history = train_classifier(classifier, new_model, x_train, y_train, epochs, batch_size, validation_data,
                                          True)
    val_mse = history.history['val_mean_squared_error']
    new_mse, new_epochs = min(val_mse), np.argmin(val_mse) + 1

    # Update the saved model
    if (not best_model) or (new_mse < best_mse):
        best_model = new_model
        best_info = new_info
        best_mse = new_mse
        best_epochs = new_epochs

# Save the model
save_file = 'data/model/random_trained_' + classifier + '_' + method + '_' + language
save_classifier(classifier, best_model, save_file)

# Perform a prediction
y_pred = predict_classifier(best_model, x_test)

# Compute the performance
print("RMSE: {}".format(evaluate_classifier("RMSE", y_test, y_pred)))
print("Accuracy: {}".format(evaluate_classifier("multi_accuracy", y_test, y_pred)))
print("Binary accuracy: {}".format(evaluate_classifier("binary_accuracy", y_test, y_pred)))

print("Best model information: {}".format(best_info))
print("MSE: {}".format(best_mse))
print("Epochs: {}".format(best_epochs))
