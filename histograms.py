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

from features import preprocess_tokenize
import os

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

print('en', np.mean(en), 'fr', np.mean(fr))
# en 3.7348151 fr 3.4951324

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


plt.show()
'''


def diff_distributions(a, b, percentile=5):
    # Align the supports
    lower_a, upper_a = np.percentile(a, [percentile, 100 - percentile])
    lower_b, upper_b = np.percentile(b, [percentile, 100 - percentile])
    inlier_a = a[np.where((a >= lower_a) & (a <= upper_a))]
    inlier_b = b[np.where((b >= lower_b) & (b <= upper_b))]
    min_a, max_a = np.min(inlier_a), np.max(inlier_a)
    min_b, max_b = np.min(inlier_b), np.max(inlier_b)
    range = min(min_a, min_b), max(max_a, max_b)

    # Compute the histograms with this given range
    dist_a, bins = np.histogram(inlier_a, bins=1000, range=range)
    dist_b, _ = np.histogram(inlier_b, bins=1000, range=range)
    dist_a = dist_a / np.sum(dist_a)
    dist_b = dist_b / np.sum(dist_b)

    print(bins)
    plt.plot(bins[:-1], dist_a - dist_b, label='b')
    plt.show()
    # integral_a = np.sum(np.multiply((dist_a), bins[:-1] / len(bins)))
    integral_terms = np.multiply((dist_a - dist_b), bins[:-1])
    print(integral_terms)
    plt.plot(bins[:-1], integral_terms, label='b')
    plt.show()
    integral_diff = np.sum(integral_terms) / len(bins)
    return integral_diff


# a = np.random.randn(10000)
# b = np.random.randn(10000)
# diff = diff_distributions(a, b)
# print(diff)

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
'''
# whole = []
# for i, file in enumerate(os.listdir('data/twitter/toto')):
#     # str_list = list(filter(None, str_list))
#     text = []
#     # if i > 2:
#     #     break
#     with open('data/twitter/toto/' + file, 'r', encoding="utf-8") as f:
#         file_tweets_list = f.readlines()
#         for i in range(len(file_tweets_list)):
#             file_tweets_list[i] = (file_tweets_list[i].replace("\n", "")).split(',', 2)
#             if len(file_tweets_list[i]) == 2:
#                 file_tweets_list[i] = file_tweets_list[i][1]
#             else:
#                 file_tweets_list[i] = ' '
#         # print(file_tweets_list)
#         text.extend(file_tweets_list)
#         # print(text)
#     text = np.asarray(text)
#     batch = preprocess_tokenize(text, langage='en')
#     whole.extend(batch)
#
# whole = list(filter(None, whole))
# lengths = [len(sentence) for sentence in whole]
# sns.distplot(lengths)
# plt.show()