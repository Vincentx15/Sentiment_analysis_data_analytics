from classifier import *
from features import load_features
from statistics import mode

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from textblob import TextBlob
import numpy as np
import seaborn as sns

'''
#  Plot data distribution
data_en = pd.read_csv('data/raw_csv/imdb.csv')
reviews_en = data_en['review']
lengths_en = [len(review.split()) for review in reviews_en]
ratings_en = data_en['rating']/2

data_fr = pd.read_csv('data/raw_csv/allocine.csv')
reviews_fr = data_fr['review']
lengths_fr = [len(review.split()) for review in reviews_fr]
ratings_fr = data_fr['rating']

sns.distplot(lengths_en, hist=False, label="En")
sns.distplot(lengths_fr, hist=False, label="Fr")
plt.xlim(0, 600)
plt.savefig('length.pdf')
plt.show()

sns.distplot(ratings_en, hist=True, kde=False, label="En")
sns.distplot(ratings_fr, hist=True, kde=False, label="Fr")
plt.savefig('ratings.pdf')
plt.show()
'''


def generate_mask(n, rate):
    """

    :param n: len of the mask
    :param rate: rate of ones
    :return: boolean mask example : generate mask(5,2/5) = [0,1,1,0,0]
    """
    ones = int(n * rate)
    mask = np.array([0] * (n - ones) + [1] * ones, dtype=bool)
    np.random.shuffle(mask)
    return mask


# mask = generate_mask(5, 2 / 5)
# mask = generate_mask(5, 2 / 5)
# print(mask)

def test_stable(distribution, rates=0.7, savefig=None):
    """
    test if a distribution is stable by plotting successive subsets ditributions
    :param: rates : iterable or scalar
    :return:
    """

    distribution = np.asarray(distribution)
    print(np.mean(distribution))
    n = len(distribution)
    sns.distplot(distribution, hist=False, label='Full', kde_kws={"lw": 4})

    # If iterable loop otherwise just one go
    try:
        subsets = [(rate, distribution[generate_mask(n, rate)]) for rate in rates]
        for rate, subset in subsets:
            sns.distplot(subset, hist=False, label=round(rate, 3))
    except TypeError:
        subset = distribution[generate_mask(n, rates)]
        sns.distplot(subset, hist=False)
    if savefig:
        plt.savefig(savefig)
    plt.show()

'''
#   Plot the wiki distributions
en = np.load('data/wikipedia/results/en.npy')
fr = np.load('data/wikipedia/results/fr.npy')

for rate in np.linspace(0.1, 0.9, 5):
    test_stable(en, rate)
test_stable(fr, [0.001, 0.01, 0.1, 0.9])
test_stable(en, [0.001, 0.01, 0.1, 0.9])

test_stable(fr, [0.001, 0.01, 0.1, 0.9], savefig='wiki_fr.pdf')
test_stable(en, [0.001, 0.01, 0.1, 0.9], savefig='wiki_en.pdf')
'''



'''
language = 'en'

# Compute the true values
*_, y_train, y_test = load_features(language, 'bow')
y_true = np.concatenate((y_train, y_test), axis=0)

classifiers = ['SVM', 'NN', 'LSTM']
methods = ['bow', 'bow', 'we']
y_pred = []

# for i in range(3):
#     print('')
#     print("{}/3...".format(i+1))
#     method = methods[i]
#     classifier = classifiers[i]
#     x_train, x_test, _, _ = load_features(language, method)
#     load_file = 'data/model/trained_' + classifier + '_' + method + '_' + language
#     model = load_classifier(classifier, load_file)
#     y_pred_train = predict_classifier(model, x_train)
#     y_pred_test = predict_classifier(model, x_test)
#     y_pred.append(np.concatenate((y_pred_train, y_pred_test), axis=0))

print("Textblob...")

data_en = pd.read_csv('data/raw_csv/imdb.csv')
reviews_en = data_en['review']
ratings_en = data_en['rating']
results_en = [TextBlob(review).sentiment[0] for review in reviews_en]
results_en = [2.5 * (res + 1) for res in results_en]
results_en = np.array(results_en)

sns.distplot(y_true, hist=False, kde_kws={"color": "b", "lw": 2, "label": "True"})
for i in range(1, 3):
    y_pred = np.load('data/saved_distributions/distributions_' + language + '_' + str(i) + '.npy')
    sns.distplot(y_pred, hist=False, kde_kws={"label": classifiers[i]})

sns.distplot(results_en, hist=False, kde_kws={"label": 'Textblob en'})

# data_fr = pd.read_csv('data/raw_csv/allocine.csv')
# reviews_fr = data_fr['review']
# ratings_fr = data_fr['rating']
# results_fr = [TextBlob(review).sentiment[0] for review in reviews_fr]
# results_fr = [2.5 * (res + 1) for res in results_fr]
# results_fr = np.array(results_fr)
# sns.distplot(results_fr, hist=False, kde_kws={"label": 'Textblob Fr'})
<<<<<<< HEAD
=======

>>>>>>> ca8d7721331165279ca516453cd8e800bd781352

plt.show()
'''

'''
distrib_en = np.load('data/wikipedia/results/en.npy')
distrib_fr = np.load('data/wikipedia/results/fr.npy')

print('en', np.mean(distrib_en), 'fr', np.mean(distrib_fr))
# en 3.7348151 fr 3.4951324


sns.distplot(distrib_en, hist=False, kde_kws={"color": "b", "lw": 2, "label": "Fr"})
sns.distplot(distrib_fr, hist=False, kde_kws={"color": "r", "lw": 2, "label": "En"})
plt.show()
'''


# Histogram of the length of the reviews
from features import *

method = 'we'
langages = ['en', 'fr']
csv_files = ['data/raw_csv/imdb.csv', 'data/raw_csv/allocine.csv']

for i in range(0,1):
    langage = langages[i]
    csv_file = csv_files[i]

    data = pd.read_csv(csv_file)
    text = data['review'].values
    labels = data['rating'].values

    # split data
    raw_train_data, raw_test_data, train_labels, test_labels = train_test_split(text, labels,
                                                                                test_size=0.33, random_state=42)

    # Do the appropriate embedding on the text
    processed_train_data = preprocess_tokenize(raw_train_data, langage=langage)
    processed_test_data = preprocess_tokenize(raw_test_data, langage=langage)

    processed_data = processed_train_data + processed_test_data
    preprocessed_length = [len(data) for data in processed_data]

    sns.distplot(preprocessed_length)
    plt.show()

    print("Mean: {}; mode: {}".format(round(np.mean(preprocessed_length), 2), round(mode(preprocessed_length), 2)))

