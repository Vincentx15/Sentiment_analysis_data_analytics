# utils
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import gc

# we
from gensim.models import KeyedVectors as Kv
from gensim.models import FastText

# bow
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from nltk.stem.snowball import FrenchStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import scipy.sparse as sp

stop_words_fr = get_stop_words('fr')
stop_words_en = get_stop_words('en')

'''
bow embedding
'''


# def french_lemmatizer(stemmer=FrenchStemmer):
#     token_pattern = re.compile(r"(?u)\b\w\w+\b")
#     return lambda doc: list(map(stemmer.stem, token_pattern.findall(doc)))

# def english_lemmatizer(lemmatizer):
#     token_pattern = re.compile(r"(?u)\b\w\w+\b")
#     return lambda doc: list(map(lemmatizer.lemmatize, token_pattern.findall(doc)))[0]

class FrenchLemmaTokenizer(object):
    def __init__(self):
        self.wnl = FrenchStemmer()

    def __call__(self, s):
        return [self.wnl.stem(t) for t in word_tokenize(s) if t.isalpha()]


class EnglishLemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, s):
        return [self.wnl.lemmatize(t) for t in word_tokenize(s) if t.isalpha()]


def bow_features(raw_train_data, raw_test_data, langage, ngram=(1, 1), min_df=0.01, max_df=0.9):
    """
    Perform bow embedding with the given parameters, the training is only conducted on the train
    :param raw_train_data:
    :param raw_test_data:
    :param stopwords:
    :param ngram:
    :param min_df:
    :param max_df:
    :return:
    """
    if langage == 'en':
        tokenizer = EnglishLemmaTokenizer()
        # No need to modify stopwords since the lemmatization will just reduce some to some others
        stopwords = stop_words_en
    elif langage == 'fr':
        tokenizer = FrenchLemmaTokenizer()
        stopwords = [tokenizer(stopword)[0] for stopword in stop_words_fr]
    else:
        raise ValueError('Wrong language ! ')
    vectorizer = CountVectorizer(input='content ',
                                 analyzer='word',
                                 ngram_range=ngram,
                                 stop_words=stopwords,
                                 min_df=min_df,
                                 max_df=max_df,
                                 tokenizer=tokenizer)

    # # just to debug
    # analyzer = vectorizer.build_analyzer()
    # processed = [analyzer(doc) for doc in raw_train_data]
    # return processed

    train_data = vectorizer.fit_transform(raw_train_data)
    test_data = vectorizer.transform(raw_test_data)
    return train_data, test_data


# data = ["I ate a cow", 'awesome, loves it. Oh fuck it is so good', 'a']
# a = bow_features(data, data, 'en')
# return a lign of zeroes if it is empty
# data = ["J'ai mangé une vache", "Génial, j'adore. Oh putain c'est tellement bon"]
# a = bow_features(data, data, 'fr')
# print(a)


'''
we method
'''


def preprocess_tokenize(data, langage, ngram=(1, 1), min_df=0.01, max_df=0.9):
    """
    read a list of strings. return a list of list of words without stopwords
    :param data:
    :param m_df:
    :param ngram:
    :param stopwords:
    :return:
    """
    if langage == 'en':
        stopwords = stop_words_en
    elif langage == 'fr':
        stopwords = stop_words_fr
    else:
        raise ValueError('Wrong language ! ')
    vectorizer = CountVectorizer(input='content ',
                                 analyzer='word',
                                 ngram_range=ngram,
                                 stop_words=stopwords,
                                 min_df=min_df,
                                 max_df=max_df)
    analyzer = vectorizer.build_analyzer()
    processed = [analyzer(doc) if (doc not in [np.NaN, np.nan]) else [] for doc in data]
    return processed


# data = ["I ate a cow", 'awesome, loves it. Oh fuck it is so good']
# a = preprocess_tokenize(data, 'en')
# print(a)


def word_embeddings(preprocessed_data, model, length_embedding, seq_l):
    """
    Returns a feature list with for each sample the embeddings of the words in the sentence
    :param preprocessed_data: list of list of string, sentences to process
    :param model: a word embedding model
    :param length_embedding: the length of each of the embedding vectors
    :param seq_l: int, length of the sequence we want to return
    :return: list of either [] or word_embedding*seq_l
    """
    x = []
    for review in preprocessed_data:
        # Create an embedding for each sentence
        sentence_embedding = []

        for word in review:
            try:
                sentence_embedding.append(model[word])
            except KeyError:
                pass  # Do nothing for OOV items

        # Check that it is not empty
        if sentence_embedding:

            # Remove elements if necessary
            if len(sentence_embedding) >= seq_l:
                x.append(sentence_embedding[0:seq_l])

            # Perform zero-padding if necessary
            else:
                for i in range(seq_l - len(sentence_embedding)):
                    sentence_embedding.append(np.zeros(length_embedding))
                x.append(sentence_embedding)
        else:
            x.append([])

    return x


def we_features(raw_train_data, raw_test_data, langage, seq_l, ngram=(1, 1), min_df=0.01, max_df=0.9):
    """
    wrapper for word embedding features to take two sets (training and test)
    """
    print('first step')
    # Load the pretrained model
    if langage == 'en':
        fname = 'data/word_embeddings/wiki.en.vec.bin'
        bin = True
        model = Kv.load_word2vec_format(fname, binary=bin)
        length_embedding = len(model['hello'])
    elif langage == 'fr':
        fname = 'data/word_embeddings/wiki.fr.vec.bin'
        bin = True
        model = Kv.load_word2vec_format(fname, binary=bin)
        length_embedding = len(model['bonjour'])
    else:
        raise ValueError('Wrong language ! ')

    print('second step')
    # using this we, compute features
    processed_train_data = preprocess_tokenize(raw_train_data, langage=langage, ngram=ngram, min_df=min_df,
                                               max_df=max_df)
    train_data = word_embeddings(processed_train_data, model=model, length_embedding=length_embedding, seq_l=seq_l)

    print('third step')
    processed_test_data = preprocess_tokenize(raw_test_data, langage=langage, ngram=ngram, min_df=min_df, max_df=max_df)
    test_data = word_embeddings(processed_test_data, model=model, length_embedding=length_embedding, seq_l=seq_l)

    return train_data, test_data


def remove_empty(data, labels=None, method='bow'):
    """
    :data the data to clean
    :param method: string for the method
    :return: clean data and clean labels if some are provided
    """
    gc.collect()
    if method == 'we':
        iloc = []
        for id, item in enumerate(data):
            if not item:
                iloc.append(id)
        clean_data = np.asarray([x for i, x in enumerate(data) if i not in iloc])
        if labels is not None:
            clean_labels = np.delete(labels, iloc)
        else:
            return clean_data

    elif method == 'bow':
        array = np.diff(data.indptr) != 0
        clean_data = data[array]
        if labels is not None:
            clean_labels = labels[array]
        else:
            return clean_data
    else:
        raise ValueError('This is not an acceptable method !')
    return clean_data, clean_labels


def create_features(input_path, langage, save_name=False, seq_l=42, ngram=(1, 1), min_df=0.01,
                    max_df=0.9, method='we', labels_name='rating', text_column='review'):
    """
    :param input_path: path of the csv to read
    :param method: method for the embedding
    :param text_column: name of the column of the csv to use for the text items
    :param labels: name of the column to use for the labels, if none enter 0
    :return: train, test with labels
    """
    data = pd.read_csv(input_path)
    text = data[text_column].values

    # careful if no labels are provided, return a np.array of shape (len(features),)
    if labels_name:
        labels = data[labels_name].values
    else:
        labels = np.zeros(text.shape)

    # split data
    raw_train_data, raw_test_data, train_labels, test_labels = train_test_split(text, labels,
                                                                                test_size=0.33, random_state=42)

    # transform our labels to be able to hstack them and remove Nan or empty values
    train_labels = train_labels[:, np.newaxis]
    test_labels = test_labels[:, np.newaxis]

    # Do the appropriate embedding on the text
    if method == 'we':
        train_data, test_data = we_features(raw_train_data, raw_test_data, langage, seq_l, ngram=ngram, min_df=min_df,
                                            max_df=max_df)
        gc.collect()
    elif method == 'bow':
        train_data, test_data = bow_features(raw_train_data, raw_test_data, langage,
                                             ngram=ngram, min_df=min_df, max_df=max_df)
    else:
        raise ValueError('This is not an acceptable method !')

    # return the approriate data
    columns = ['text']
    if labels_name:
        train_data, train_labels = remove_empty(train_data, train_labels, method)
        print('fourth_step')
        test_data, test_labels = remove_empty(test_data, test_labels, method)

        return train_data, test_data, train_labels, test_labels

    #     columns.append('label')
    #     train = hstack((data, train_labels))
    #     test = hstack((test_data, test_labels))
    #     data = pd.DataFrame(data=train.toarray())
    #     test_data = pd.DataFrame(data=test.toarray())
    # else:
    #     data = pd.DataFrame(data, columns=columns)
    #     test_data = pd.DataFrame(test_data, columns=columns)
    # if save_name:
    #     data.to_csv('data/' + save_name)
    #     test_data.to_csv('data/' + save_name)

    train_data = remove_empty(train_data, method=method)
    test_data = remove_empty(test_data, method=method)
    return train_data, test_data


def wiki(input_path, method='we', seq_l=42, ngram=(1, 1), min_df=0.01,
         max_df=0.9):
    """
    :param input_path:
    :param method:
    """
    data = pd.read_csv(input_path)
    text_en = data['summary_en'].values
    text_fr = data['summary_fr'].values
    # print(type(text_en))
    # first = text_en[0]
    # print(first, type(first))

    # Do the appropriate embedding on the text
    if method == 'we':
        # English
        fname = 'data/word_embeddings/wiki.en.vec.bin'
        bin = True
        model = Kv.load_word2vec_format(fname, binary=bin)
        length_embedding = len(model['hello'])

        processed_en = preprocess_tokenize(text_en, langage='en', ngram=ngram, min_df=min_df,
                                           max_df=max_df)
        text_en = word_embeddings(processed_en, model=model, length_embedding=length_embedding, seq_l=seq_l)
        gc.collect()
        print('third step')
        # French
        fname = 'data/word_embeddings/wiki.fr.vec.bin'
        bin = True
        model = Kv.load_word2vec_format(fname, binary=bin)
        length_embedding = len(model['bonjour'])

        processed_fr = preprocess_tokenize(text_fr, langage='fr', ngram=ngram, min_df=min_df,
                                           max_df=max_df)
        text_fr = word_embeddings(processed_fr, model=model, length_embedding=length_embedding, seq_l=seq_l)
    else:
        raise ValueError('This is not an acceptable method !')

    text_en, text_fr = remove_empty(text_en, text_fr, method)
    text_fr, text_en = remove_empty(text_fr, text_en, method)

    return text_en, text_fr


def save_features(train_data, test_data, train_labels, test_labels, language, method):
    fname = "data/features/"
    if method == "bow":
        sp.save_npz(fname + 'train_data_bow_' + language + '.npz', train_data)
        sp.save_npz(fname + 'test_data_bow_' + language + '.npz', test_data)
        np.save(fname + 'train_labels_bow_' + language, train_labels)
        np.save(fname + 'test_labels_bow_' + language, test_labels)
        print("Data saved.")

    elif method == "we":
        np.save(fname + 'train_data_we_' + language, train_data)
        np.save(fname + 'test_data_we_' + language, test_data)
        np.save(fname + 'train_labels_we_' + language, train_labels)
        np.save(fname + 'test_labels_we_' + language, test_labels)
        print("Data saved.")
    elif method == "wiki":
        np.save(fname + 'en', train_data)
        np.save(fname + 'fr', test_data)
    else:
        raise ValueError("Wrong method.")


def load_features(language, method):
    fname = "data/features/"
    if method == "bow":
        train_data = sp.load_npz(fname + 'train_data_bow_' + language + '.npz')
        test_data = sp.load_npz(fname + 'test_data_bow_' + language + '.npz')
        train_labels = np.ravel(np.load(fname + 'train_labels_bow_' + language + '.npy'))
        test_labels = np.ravel(np.load(fname + 'test_labels_bow_' + language + '.npy'))
        if language == 'en':
            train_labels = np.divide(train_labels, 2.)
            test_labels = np.divide(test_labels, 2.)
        print("Data loaded.")
        return train_data, test_data, train_labels, test_labels

    elif method == "we":
        train_data = np.load(fname + 'train_data_we_' + language + '.npy')
        test_data = np.load(fname + 'test_data_we_' + language + '.npy')
        train_labels = np.ravel(np.load(fname + 'train_labels_we_' + language + '.npy'))
        test_labels = np.ravel(np.load(fname + 'test_labels_we_' + language + '.npy'))
        if language == 'en':
            train_labels = np.divide(train_labels, 2.)
            test_labels = np.divide(test_labels, 2.)
        print("Data loaded.")
        return train_data, test_data, train_labels, test_labels

    else:
        raise ValueError("Wrong method.")


if __name__ == '__main__':
    # method = 'bow'
    # language = 'en'
    # csv_file = 'data/raw_csv/imdb.csv'
    #
    # t1 = time.time()
    # train_data, test_data, train_labels, test_labels = create_features(csv_file, language, method=method)
    # print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
    #
    # save_features(train_data, test_data, train_labels, test_labels, language, method)
    # print(time.time() - t1)

    csv_file = 'data/wikipedia/samples.csv'

    t1 = time.time()
    en, fr = wiki(csv_file)
    print(en, fr)
    save_features(en, fr, [], [],'',method='wiki')

    # save_features(train_data, test_data, train_labels, test_labels, language, method)
    print(time.time() - t1)
