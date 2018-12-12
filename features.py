import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors as Kv
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from nltk.stem.snowball import FrenchStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import scipy.sparse as sp

stop_words_fr = get_stop_words('fr')
to_keep_fr = ["bon", "moins", "ni", "pas", "peu", "tellement", "trop", "très"]
stop_words_fr = list(set(stop_words_fr) - set(to_keep_fr))

stop_words_en = get_stop_words('en')
to_keep_en = ["aren't", "didn't", "doesn't", "don't", "isn't", "more", "most", "mustn't", "no", "nor", "not", "so",
              "too", "very", "wasn't", "weren't"]
stop_words_en = list(set(stop_words_en) - set(to_keep_en))


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


def bow_features(raw_train_data, raw_test_data, language, ngram=(1, 1), min_df=0.01, max_df=0.9):
    """
    Perform bow embedding with the given parameters, the training is only conducted on the train
    :param raw_train_data:
    :param raw_test_data:
    :param language:
    :param ngram:
    :param min_df:
    :param max_df:
    :return:
    """

    if language == 'en':
        tokenizer = EnglishLemmaTokenizer()
        stopwords = stop_words_en

    elif language == 'fr':
        tokenizer = FrenchLemmaTokenizer()
        stopwords = [tokenizer(stopword)[0] for stopword in stop_words_fr]

    else:
        raise ValueError('Wrong language ! ')

    vectorizer = CountVectorizer(input='content ', analyzer='word', ngram_range=ngram, stop_words=stopwords,
                                 min_df=min_df, max_df=max_df, tokenizer=tokenizer)

    train_data = vectorizer.fit_transform(raw_train_data)
    test_data = vectorizer.transform(raw_test_data)

    return train_data, test_data


def preprocess_tokenize(data, language, ngram=(1, 1), min_df=0.01, max_df=0.9):
    """
    Read a list of strings. return a list of list of words without stopwords
    :param data:
    :param language:
    :param ngram:
    :param min_df:
    :param max_df:
    :return:
    """

    if language == 'en':
        stopwords = stop_words_en
    elif language == 'fr':
        stopwords = stop_words_fr
    else:
        raise ValueError('Wrong language ! ')

    vectorizer = CountVectorizer(input='content ', analyzer='word', ngram_range=ngram, stop_words=stopwords,
                                 min_df=min_df, max_df=max_df)

    analyzer = vectorizer.build_analyzer()
    processed = [analyzer(doc) if (doc not in [np.NaN, np.nan]) else [] for doc in data]

    return processed


def word_embeddings(preprocessed_data, model, length_embedding, min_seq_length, max_seq_length):
    """
    Returns a feature list with for each sample the embeddings of the words in the sentence
    :param preprocessed_data: list of list of string, sentences to process
    :param model: a word embedding model
    :param length_embedding: the length of each of the embedding vectors
    :param min_seq_length: int, min length of the sequence we want to return
    :param max_seq_length: int, max length of the sequence we want to return
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
            if len(sentence_embedding) >= max_seq_length:
                sentence_embedding = np.asarray(sentence_embedding[0:max_seq_length])
                x.append(sentence_embedding)

            # Perform zero-padding if necessary
            elif len(sentence_embedding) >= min_seq_length:
                for i in range(max_seq_length - len(sentence_embedding)):
                    sentence_embedding.append(np.zeros(length_embedding))
                sentence_embedding = np.asarray(sentence_embedding)
                x.append(sentence_embedding)

            else:
                x.append([])

        else:
            x.append([])

    return x


def we_features(raw_train_data, raw_test_data, language, min_seq_length, max_seq_length, ngram=(1, 1), min_df=0.01,
                max_df=0.9):
    """
    Wrapper for word embedding features to take two sets (training and test)
    :param raw_train_data:
    :param raw_test_data:
    :param language:
    :param min_seq_length:
    :param max_seq_length:
    :param ngram:
    :param min_df:
    :param max_df:
    :return:
    """

    # Load the pretrained model
    if language == 'en':
        fname = 'data/word_embeddings/wiki.en.vec.bin'
        binary = True
        model = Kv.load_word2vec_format(fname, binary=binary)
        length_embedding = len(model['hello'])

    elif language == 'fr':
        fname = 'data/word_embeddings/wiki.fr.vec.bin'
        binary = True
        model = Kv.load_word2vec_format(fname, binary=binary)
        length_embedding = len(model['bonjour'])

    else:
        raise ValueError('Wrong language ! ')

    processed_train_data = preprocess_tokenize(raw_train_data, language=language, ngram=ngram, min_df=min_df,
                                               max_df=max_df)
    train_data = word_embeddings(processed_train_data, model=model, length_embedding=length_embedding,
                                 min_seq_length=min_seq_length, max_seq_length=max_seq_length)

    processed_test_data = preprocess_tokenize(raw_test_data, language=language, ngram=ngram, min_df=min_df,
                                              max_df=max_df)
    test_data = word_embeddings(processed_test_data, model=model, length_embedding=length_embedding,
                                min_seq_length=min_seq_length, max_seq_length=max_seq_length)

    return train_data, test_data


def remove_empty(data, labels=None, list_labels=None, method='bow'):
    """
    Remove empty lists in the data
    :param data: the data to clean
    :param labels: string that indicate the kind of labels provided, if wiki, treats the second columns as embeddings
    :param list_labels: the actual iterable label
    :param method: string for the method
    :return: clean data and clean labels if some are provided
    """

    if method == 'we':
        iloc = []
        for id, item in enumerate(data):
            if len(item) == 0:
                iloc.append(id)
        if labels == 'wiki':
            iloc_label = []
            for id, item in enumerate(list_labels):
                # print(item)
                if len(item) == 0:
                    iloc_label.append(id)
            iloc = set(iloc)
            iloc_label = set(iloc_label)
            iloc_selected = list(iloc_label.union(iloc))
            clean_data = np.delete(np.asarray(data), iloc_selected)
            clean_data = np.stack(clean_data)
            clean_labels = np.delete(np.asarray(list_labels), iloc_selected)
            clean_labels = np.stack(clean_labels)

        elif labels is not None:
            clean_data = np.asarray([x for i, x in enumerate(data) if i not in iloc])
            clean_data = np.stack(clean_data)
            clean_labels = np.delete(np.asarray(list_labels), iloc)
            clean_labels = np.stack(clean_labels)
        else:
            clean_data = np.asarray([x for i, x in enumerate(data) if i not in iloc])
            clean_data = np.stack(clean_data)
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


def create_features(database, language, method, min_seq_length=5, max_seq_length=20, ngram=(1, 1), min_df=0.01,
                    max_df=0.9, labels_name='rating', text_column='review', files_nb=0, max_tweets=0, query='',
                    extended=False):
    """
    Creates features
    :param database: path of the csv to read
    :param language:
    :param min_seq_length:
    :param max_seq_length:
    :param ngram:
    :param min_df:
    :param max_df:
    :param method: method for the embedding
    :param labels_name: name of the column to use for the labels, if none enter 0
    :param text_column: name of the column of the csv to use for the text items
    :param files_nb:
    :param max_tweets:
    :param query:
    :param extended:
    :return: train, test with labels
    """

    if database == 'imdb':
        data = pd.read_csv('data/raw_csv/imdb.csv')

    elif database == 'allocine':
        data = pd.read_csv('data/raw_csv/allocine.csv')

    elif database == 'wikipedia':
        fname = 'data/raw_csv/wikipedia.csv'
        text_en, text_fr = wiki(fname, method=method, min_seq_length=min_seq_length, max_seq_length=max_seq_length,
                                ngram=ngram, min_df=min_df, max_df=max_df)
        if language == 'en':
            return text_en
        elif language == 'fr':
            return text_fr

    elif database == 'twitter':
        input_folder_path = 'data/twitter'
        data = twitter(input_folder_path, files_nb, max_tweets, query, language, extended=extended, method=method,
                       min_seq_length=min_seq_length, max_seq_length=max_seq_length, ngram=ngram, min_df=min_df,
                       max_df=max_df)
        return data

    else:
        raise ValueError("Wrong database.")

    text = data[text_column].values

    # careful if no labels are provided, return a np.array of shape (len(features),)
    if labels_name:
        labels = data[labels_name].values
    else:
        labels = np.zeros(text.shape)

    # split data
    raw_train_data, raw_test_data, train_labels, test_labels = train_test_split(text, labels,
                                                                                test_size=0.25, random_state=42)

    # transform our labels to be able to hstack them and remove Nan or empty values
    train_labels = train_labels[:, np.newaxis]
    test_labels = test_labels[:, np.newaxis]

    # Do the appropriate embedding on the text
    if method == 'we':
        train_data, test_data = we_features(raw_train_data, raw_test_data, language, min_seq_length=min_seq_length,
                                            max_seq_length=max_seq_length, ngram=ngram, min_df=min_df, max_df=max_df)
    elif method == 'bow':
        train_data, test_data = bow_features(raw_train_data, raw_test_data, language,
                                             ngram=ngram, min_df=min_df, max_df=max_df)
    else:
        raise ValueError('This is not an acceptable method !')

    # return the approriate data
    if labels_name:
        train_data, train_labels = remove_empty(train_data, labels_name, train_labels, method)
        test_data, test_labels = remove_empty(test_data, labels_name, test_labels, method)

        if 'database' == 'imdb':
            train_labels = np.divide(train_labels, 2.0)
            test_labels = np.divide(test_labels, 2.0)

        return train_data, test_data, train_labels, test_labels

    train_data = remove_empty(train_data, method=method)
    test_data = remove_empty(test_data, method=method)
    return train_data, test_data


def wiki(input_path, method='we', min_seq_length=5, max_seq_length=20, ngram=(1, 1), min_df=0.01, max_df=0.9):
    """
    Functions for wikipedia features
    :param input_path:
    :param method:
    :param min_seq_length:
    :param max_seq_length:
    :param ngram:
    :param min_df:
    :param max_df:
    :return:
    """

    data = pd.read_csv(input_path)
    text_en = data['summary_en'].values
    text_fr = data['summary_fr'].values

    # Do the appropriate embedding on the text
    if method == 'we':
        # English
        fname = 'data/word_embeddings/wiki.en.vec.bin'
        bin = True
        model = Kv.load_word2vec_format(fname, binary=bin)
        length_embedding = len(model['hello'])

        processed_en = preprocess_tokenize(text_en, language='en', ngram=ngram, min_df=min_df,
                                           max_df=max_df)
        text_en = word_embeddings(processed_en, model=model, length_embedding=length_embedding,
                                  min_seq_length=min_seq_length, max_seq_length=max_seq_length)
        # French
        fname = 'data/word_embeddings/wiki.fr.vec.bin'
        bin = True
        model = Kv.load_word2vec_format(fname, binary=bin)
        length_embedding = len(model['bonjour'])

        processed_fr = preprocess_tokenize(text_fr, language='fr', ngram=ngram, min_df=min_df,
                                           max_df=max_df)
        text_fr = word_embeddings(processed_fr, model=model, length_embedding=length_embedding,
                                  min_seq_length=min_seq_length, max_seq_length=max_seq_length)
        print('French done')
    else:
        raise ValueError('This is not an acceptable method !')

    text_en, text_fr = remove_empty(data=text_en, labels='wiki', list_labels=text_fr, method='we')

    return text_en, text_fr


def twitter(input_folder_path, files_nb, max_tweets, query, language, extended=False, method='we', min_seq_length=5,
            max_seq_length=20, ngram=(1, 1), min_df=0.01, max_df=0.9):
    """
    Create the features for twitter files
    :param input_folder_path:
    :param files_nb:
    :param max_tweets:
    :param query:
    :param language:
    :param extended:
    :param method:
    :param min_seq_length:
    :param max_seq_length:
    :param ngram:
    :param min_df:
    :param max_df:
    :return:
    """

    if extended:
        str_extended = '_extended'
    else:
        str_extended = ''
    fnames = [input_folder_path + '/twitter_server_' + str(i) + '__' + language + '_' + str(
        max_tweets) + '_' + query + str_extended + '.txt' for i in range(1, files_nb + 1)]

    text = []
    for fname in fnames:
        with open(fname, 'r', encoding="utf-8") as f:
            file_tweets_list = f.readlines()
            for i in range(len(file_tweets_list)):
                file_tweets_list[i] = (file_tweets_list[i].replace("\n", "")).split(',', 2)
                if len(file_tweets_list[i]) == 2:
                    file_tweets_list[i] = file_tweets_list[i][1]
                else:
                    file_tweets_list[i] = ' '
            text.extend(file_tweets_list)

    text = np.asarray(text)
    labels = np.zeros(text.shape)

    # split data
    raw_train_data, raw_test_data, _, _ = train_test_split(text, labels, test_size=0.33, random_state=42)

    # Do the appropriate embedding on the text
    if method == 'we':
        train_data, test_data = we_features(raw_train_data, raw_test_data, language, min_seq_length=min_seq_length,
                                            max_seq_length=max_seq_length, ngram=ngram, min_df=min_df, max_df=max_df)
    elif method == 'bow':
        train_data, test_data = bow_features(raw_train_data, raw_test_data, language, ngram=ngram, min_df=min_df,
                                             max_df=max_df)
    else:
        raise ValueError('This is not an acceptable method !')

    train_data = remove_empty(train_data, method=method)
    test_data = remove_empty(test_data, method=method)
    data = np.concatenate((train_data, test_data), axis=0)
    return data


def save_features(database, language, method, train_data=None, test_data=None, train_labels=None, test_labels=None,
                  query='all', extended=False):
    """
    Save the features
    :param database:
    :param language:
    :param method:
    :param train_data:
    :param test_data:
    :param train_labels:
    :param test_labels:
    :param query:
    :param extended:
    :return:
    """
    fname = "data/features/" + database + '_'
    if database in ['allocine', 'imdb']:
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

        else:
            raise ValueError("Wrong method.")

    elif database == 'wikipedia':
        if method == "bow":
            sp.save_npz(fname + 'test_data_bow_' + language + '.npz', test_data)
            print("Data saved.")

        elif method == "we":
            np.save(fname + 'test_data_we_' + language, test_data)
            print("Data saved.")

        else:
            raise ValueError("Wrong method.")

    elif database == 'twitter':
        if extended:
            str_extended = 'extended'
        else:
            str_extended = ''

        if method == "bow":
            sp.save_npz(fname + 'test_data_bow_' + language + '_' + query + '_' + str_extended + '.npz', test_data)
            print("Data saved.")

        elif method == "we":
            np.save(fname + 'test_data_we_' + language + '_' + query + '_' + str_extended, test_data)
            print("Data saved.")

        else:
            raise ValueError("Wrong method.")

    else:
        raise ValueError("Wrong database.")


def load_features(database, language, method, query='all', extended=False):
    """
    Load the features
    :param database:
    :param language:
    :param method:
    :param query:
    :param extended:
    :return:
    """
    fname = "data/features/" + database + '_'
    if database in ['allocine', 'imdb']:
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

    elif database == 'wikipedia':
        if method == "bow":
            test_data = sp.load_npz(fname + 'test_data_bow_' + language + '.npz')
            print("Data loaded.")
            return test_data

        elif method == "we":
            test_data = np.load(fname + 'test_data_we_' + language + '.npy')
            print("Data loaded.")
            return test_data

    elif database == 'twitter':
        if extended:
            str_extended = 'extended'
        else:
            str_extended = ''

        if method == "bow":
            test_data = sp.load_npz(fname + 'test_data_bow_' + language + '_' + query + '_' + str_extended + '.npz')
            print("Data loaded.")
            return test_data

        elif method == "we":
            test_data = np.load(fname + 'test_data_we_' + language + '_' + query + '_' + str_extended + '.npy')
            print("Data loaded.")
            return test_data

    else:
        raise ValueError("Wrong database")


def save_prediction(y_pred, database, classifier, method, language, query='all', extended=False):
    """
    Save a prediction
    :param database:
    :param classifier:
    :param method:
    :param language:
    :param query:
    :param extended:
    :return:
    """

    if database in ['allocine', 'imdb', 'wikipedia']:
        np.save('data/predictions/' + database + '_' + classifier + '_' + method + '_' + language, y_pred)

    elif database == 'twitter':
        if extended:
            str_extended = 'extended'
        else:
            str_extended = ''
        np.save('data/predictions/' + database + '_' + classifier + '_' + method + '_' + language + '_' + query + '_' +
                str_extended, y_pred)

    else:
        raise ValueError("Wrong database.")


if __name__ == '__main__':
    '''
    database = 'allocine'
    language = 'fr'
    method = 'we'
    min_seq_length = 5
    max_seq_length = 20

    train_data, test_data, train_labels, test_labels = create_features(database, language, method,
                                                                       min_seq_length=min_seq_length,
                                                                       max_seq_length=max_seq_length)
    '''
    '''
    database = 'wikipedia'
    language = 'fr'
    method = 'we'
    min_seq_length = 5
    max_seq_length = 20

    test_data = create_features(database, language, method,
                                min_seq_length=min_seq_length,
                                max_seq_length=max_seq_length)
    save_features(database, language, method, test_data=test_data)
    '''

    database = 'twitter'
    # language = 'fr'
    # query = 'giletsjaunes'
    language = 'en'
    query = 'yellowvest'
    method = 'we'
    min_seq_length = 5
    max_seq_length = 20
    files_nb = 1
    max_tweets = 2000
    extended = True

    test_data = create_features(database, language, method, query=query, files_nb=files_nb, max_tweets=max_tweets,
                                extended=extended, min_seq_length=min_seq_length, max_seq_length=max_seq_length)
    save_features(database, language, method, test_data=test_data, query=query, extended=extended)
