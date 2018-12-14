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

    sns.distplot(lengths_en, hist=True, label="Allociné", bins=np.linspace(0, 600, 20))
    sns.distplot(lengths_fr, hist=True, label="Imdb", bins=np.linspace(0, 600, 20))
    plt.xlim(0, 600)
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


def plot_pred_and_labels(language):
    """
    Plot the distributions of the true values and the predictions
    :param language: string
    :return:
    """

    data_en = pd.read_csv('data/raw_csv/imdb.csv')
    ratings_en = data_en['rating'] / 2

    data_fr = pd.read_csv('data/raw_csv/allocine.csv')
    ratings_fr = data_fr['rating']

    pred_en = np.load('data/predictions/imdb_LSTM_we_en.npy')
    pred_fr = np.load('data/predictions/allocine_LSTM_we_fr.npy')

    if language == 'en':
        sns.distplot(ratings_en, hist=True, kde=False, label="Allociné", norm_hist=True)
        sns.distplot(pred_en)

    elif language == 'fr':
        sns.distplot(ratings_fr, hist=True, kde=False, label="Imdb", norm_hist=True)
        sns.distplot(pred_fr)

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
    """
    Compute a scalar relevant for comparing distributions by computing integral[ (a(x)-b(x))**1/3 x dx]
    Aligns the supports by outliers filtering and zero padding
    :param a: a distribution as a np array of observed values
    :param b: a distribution as a np array of observed values
    :param percentile: filtering parameter for the outliers
    :return: the computed integrals
    """
    # Align the supports
    lower_a, upper_a = np.percentile(a, [percentile, 100 - percentile])
    lower_b, upper_b = np.percentile(b, [percentile, 100 - percentile])
    inlier_a = a[np.where((a >= lower_a) & (a <= upper_a))]
    inlier_b = b[np.where((b >= lower_b) & (b <= upper_b))]
    min_a, max_a = np.min(inlier_a), np.max(inlier_a)
    min_b, max_b = np.min(inlier_b), np.max(inlier_b)
    range = min(min_a, min_b), max(max_a, max_b)

    # Compute the normalised histograms with this given range
    dist_a, bins = np.histogram(inlier_a, bins=1000, range=range)
    dist_b, _ = np.histogram(inlier_b, bins=1000, range=range)
    dist_a = dist_a / np.sum(dist_a)
    dist_b = dist_b / np.sum(dist_b)

    # plt.plot(bins[:-1], dist_a - dist_b, label='b')
    # plt.show()

    # Compute the integral
    def cubic_root(x):
        if 0 <= x: return x ** (1. / 3.)
        return -(-x) ** (1. / 3.)

    integral_terms = np.multiply(np.array([cubic_root(x) for x in (dist_a - dist_b)]), bins[:-1])
    # print(integral_terms)
    # plt.plot(bins[:-1], integral_terms, label='integrated function')
    # plt.show()
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


def read_twitter_data(path):
    """
    Length distributions of sentences in twitter after tokenization but before word embeddings
    :return:
    """
    whole = []
    for i, file in enumerate(os.listdir()):
        tweets = []
        # if i > 2:
        #     break
        with open('data/twitter/toto/' + file, 'r', encoding="utf-8") as f:
            file_tweets_list = f.readlines()
            for i in range(len(file_tweets_list)):
                file_tweets_list[i] = (file_tweets_list[i].replace("\n", "")).split(',', 2)
                if len(file_tweets_list[i]) == 2:
                    file_tweets_list[i] = file_tweets_list[i][1]
                else:
                    file_tweets_list[i] = ' '
            tweets.extend(file_tweets_list)
        tweets = np.asarray(tweets)
        batch = preprocess_tokenize(tweets, langage='en')
        whole.extend(batch)

    whole = list(filter(None, whole))
    lengths = [len(sentence) for sentence in whole]
    sns.distplot(lengths)
    plt.show()


if __name__ == '__main__':
    pass

    mean_en = 3.34493
    mean_fr = 3.44599

    # plot_lengths_movies_dataset()
    # plot_ratings_movies_dataset()

    # plot_wiki_stability('data/predictions/wikipedia_LSTM_we_en.npy')
    # plot_wiki_stability('data/predictions/wikipedia_LSTM_we_fr.npy')

    # plot_wiki_distribution('data/predictions/wikipedia_LSTM_we_en.npy', 'data/predictions/wikipedia_LSTM_we_fr.npy')

    # plot_pred_and_labels('en')
    # plot_pred_and_labels('fr')

    # read_twitter_data('data/twitter/toto')

    # plot_twitter_with_query('data/twitter/results/old_LSTM_we_en.npy',
    #                         'data/twitter/results/old_yellowvest_LSTM_we_en.npy', 3.73)
    # plot_twitter_with_query('data/twitter/results/old_LSTM_we_fr.npy',
    #                         'data/twitter/results/old_giletsjaunes_LSTM_we_fr.npy', 3.49)
    # sns.distplot(np.load('data/predictions/twitter_LSTM_we_en_yellowvest_extended.npy'))

    data_fr = np.load('data/predictions/twitter_LSTM_we_fr_Macron_extended.npy')-mean_fr
    sns.distplot(data_fr)
    print(np.mean(data_fr))

    # data_en = np.load('data/predictions/twitter_LSTM_we_en_all_extended.npy') - mean_en
    # sns.distplot(data_en)
    # print(np.mean(data_en))
    #
    # plt.show
    test_stable(data_fr,[0.001, 0.01, 0.1])
