from classifier import *
from features import load_features
from statistics import mode
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
from features import preprocess_tokenize
import os



def plot_ratings_movies_dataset():
    """
    Plot the ratings of imdb and allocine
    :return: /
    """

    data_en = pd.read_csv('data/raw_csv/imdb.csv')
    ratings_en = data_en['rating'] / 2

    data_fr = pd.read_csv('data/raw_csv/allocine.csv')
    ratings_fr = data_fr['rating']

    sns.distplot(ratings_en, hist=True, kde=False, label="Allociné", norm_hist=True)
    sns.distplot(ratings_fr, hist=True, kde=False, label="Imdb", norm_hist=True)
    plt.show()


def plot_lengths_movies_dataset():
    """
    Plot the lengths of imdb and allocine
    :return: /
    """

    data_en = pd.read_csv('data/raw_csv/imdb.csv')
    reviews_en = data_en['review']
    lengths_en = [len(review.split()) for review in reviews_en]

    data_fr = pd.read_csv('data/raw_csv/allocine.csv')
    reviews_fr = data_fr['review']
    lengths_fr = [len(review.split()) for review in reviews_fr]

    sns.distplot(lengths_en, hist=True, label="Allociné")
    sns.distplot(lengths_fr, hist=True, label="Imdb")
    plt.xlim(0, 600)
    plt.savefig('length.pdf')
    plt.show()


def plot_wiki_stability(fname):
    """
    Plot the wiki distributions
    :param fname: string, file to load
    :return: /
    """

    wiki = np.load(fname)
    test_stable(wiki, [0.001, 0.01, 0.1, 0.9])


def plot_wiki_distribution(en_fname, fr_fname):
    """
    Print wiki distributions
    :param en_fname: string
    :param fr_fname: string
    :return: /
    """

    distrib_en = np.load(en_fname)
    distrib_fr = np.load(fr_fname)

    print('Means: en:', np.mean(distrib_en), '; fr:', np.mean(distrib_fr))

    sns.distplot(distrib_en, hist=True, kde_kws={"color": "b", "lw": 2, "label": "Fr"})
    sns.distplot(distrib_fr, hist=True, kde_kws={"color": "r", "lw": 2, "label": "En"})
    plt.show()


def plot_pred_and_labels(language, plot_labels, classifiers):
    """
    Plot the distributions of the true values and the predictions
    :param language: string
    :param plot_labels: bool
    :param classifiers: list of string
    :return:
    """

    if plot_labels:
        *_, y_train, y_test = load_features(language, 'we')
        y_true = np.concatenate((y_train, y_test), axis=0)
        sns.distplot(y_true, hist=True, kde_kws={"color": "b", "lw": 2, "label": "True"})

    for i in range(0, 3):
        y_pred = np.load('data/saved_distributions/distributions_' + language + '_' + str(i) + '.npy')
        sns.distplot(y_pred, hist=False, kde_kws={"label": classifiers[i]})

    plt.show()


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


def test_stable(distribution, rates=0.7, savefig=None, print_mean=False):
    """
    Test if a distribution is stable by plotting successive subsets distributions
    :param: rates : iterable or scalar
    :return: /
    """

    distribution = np.asarray(distribution)
    if print_mean:
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

def plot_twitter_with_query(fname1, fname2, mean_calibration):
    """
    Plot twitter's prediction with a query
    :param fname1: string
    :param fname2: string
    :param mean_calibration: float
    :return: /
    """

    plot_1 = np.load(fname1) - mean_calibration
    plot_2 = np.load(fname2) - mean_calibration

    print("Means: {} & {}".format(np.mean(plot_1), np.mean(plot_1)))

    sns.distplot(plot_1)
    sns.distplot(plot_2)



if __name__ == '__main__':
    # plot_lengths_movies_dataset()
    # plot_ratings_movies_dataset()

    # plot_wiki_stability('data/wikipedia/en_results_LSTM_we_en.npy')
    # plot_wiki_stability('data/wikipedia/fr_results_LSTM_we_fr.npy')

    # ''' Means: en: 3.7348151; fr: 3.4951324'''
    # plot_wiki_distribution('data/wikipedia/en_results_LSTM_we_en.npy', 'data/wikipedia/fr_results_LSTM_we_fr.npy')


    # print("Mean: {}; mode: {}".format(round(np.mean(preprocessed_length), 2), round(mode(preprocessed_length), 2)))

    # plot_pred_and_labels('en', True, ['SVM', 'NN', 'LSTM'])
    # plot_pred_and_labels('fr', True, ['SVM', 'NN', 'LSTM'])

    # plot_twitter_with_query('data/twitter/results/old_LSTM_we_en.npy',
    #                         'data/twitter/results/old_yellowvest_LSTM_we_en.npy', 3.73)
    plot_twitter_with_query('data/twitter/results/old_LSTM_we_fr.npy',
                            'data/twitter/results/old_giletsjaunes_LSTM_we_fr.npy', 3.49)

# Length distributions of twitter
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