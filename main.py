from features import *
from classifier import *

import numpy as np

# TODO
# Read and extract the train_data
# Preprocessing (removing stop words, etc)
# Obtain train_data similar to:
# List of list of words
train_data = [['amazing', 'movie', 'genius', 'director', 'incredible', 'good'],
              ['perfect', 'plot', 'interesting', 'actors'],
              ['shitty', 'very', 'bad', 'boring'],
              ['not', 'recommend', 'stupid', 'bad', 'poor', 'crap']]
test_data = [['perfect', 'movie', 'great', 'director', 'interesting'],
             ['poorly', 'written', 'bad', 'review', 'boring']]
# numpy array
y_train = np.asarray([[1.], [1.], [0.], [0.]])
y_test = np.asarray([[1.], [0.]])


# Extract word embedding
embedding_fname = 'data/word_embeddings/GoogleNews-vectors-negative300.bin'
binary = True
seq_len = 5

x_train = word_embeddings(embedding_fname, binary, train_data, seq_len)
x_test = word_embeddings(embedding_fname, binary, test_data, seq_len)

feature_len = x_test[0].shape[1]

# Create a classifier
classifier = "LSTM"
measure = "MSE"
file_name = "data/model/first_model"
epochs = 4
batch_size = 1
parameters['LSTM']['input_shape'] = (seq_len, feature_len)
parameters['LSTM']['cells'] = 1
parameters['LSTM']['units'] = [1]
parameters['LSTM']['return_sequences'] = [False]

# model = create_classifier(classifier, parameters)
model = load_classifier(classifier, file_name)
model = train_classifier(model, x_train, y_train, epochs, batch_size)
print(predict_classifier(model,x_test))
save_classifier(classifier, model, file_name)