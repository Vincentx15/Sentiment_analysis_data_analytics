import numpy as np
from gensim.models import KeyedVectors as Kv
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from nltk.stem.snowball import FrenchStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words_fr = get_stop_words('fr')
stop_words_en = get_stop_words('en')


'''
bow embedding
'''


# def french_lemmatizer(stemmer=FrenchStemmer):
#     token_pattern = re.compile(r"(?u)\b\w\w+\b")
#     return lambda doc: list(map(stemmer.stem, token_pattern.findall(doc)))


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


# def english_lemmatizer(lemmatizer):
#     token_pattern = re.compile(r"(?u)\b\w\w+\b")
#     return lambda doc: list(map(lemmatizer.lemmatize, token_pattern.findall(doc)))[0]


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


# data = ["I ate a cow", 'awesome, loves it. Oh fuck it is so good']
# a = bow_features(data, data, 'en')
# print(a)
#
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
    if langage == 'fr':
        stopwords = stop_words_fr
    else:
        ValueError('Wrong language ! ')
    vectorizer = CountVectorizer(input='content ',
                                 analyzer='word',
                                 ngram_range=ngram,
                                 stop_words=stopwords,
                                 min_df=min_df,
                                 max_df=max_df)
    analyzer = vectorizer.build_analyzer()
    processed = [analyzer(doc) for doc in data]
    return processed


# data = ["I ate a cow", 'awesome, loves it. Oh fuck it is so good']
# a = preprocess_tokenize(data, 'en')
# print(a)


def word_embeddings(fname, b, d, seq_l):
    """
    Returns a feature array with for each sample the embeddings of the words in the sentence
    :param fname: string, file name of the pretrained embedding
    :param b: bool, is the file binary or not
    :param d: list of list of string, sentences to process
    :param seq_l: int, length of the sequence we want to return
    :return:
    """
    # Load the pretrained model
    model = Kv.load_word2vec_format(fname, binary=b)
    feat_l = len(model['hello'])

    x = []
    for sentence in d:
        # Create an embedding for each sentence
        sentence_embedding = []

        for word in sentence:
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
                    sentence_embedding.append(np.zeros(feat_l))
                x.append(sentence_embedding)

    return np.asarray(x)


def we_features(raw_train_data, raw_test_data, langage, fname, b, seq_l, ngram=(1, 1), min_df=0.01, max_df=0.9):
    """
    wrapper for word embedding features to take two sets (training and test)
    """
    processed_train_data = preprocess_tokenize(raw_train_data, langage=langage, ngram=ngram, min_df=min_df,
                                               max_df=max_df)
    train_data = word_embeddings(fname, b, processed_train_data, seq_l)
    processed_test_data = preprocess_tokenize(raw_test_data, langage=langage, ngram=ngram, min_df=min_df, max_df=max_df)
    test_data = word_embeddings(fname, b, processed_test_data, seq_l)
    return train_data, test_data


def create_features(input_path, langage, save_name=False, fname='toto', b=True, seq_l=42, ngram=(1, 1), min_df=0.01,
                    max_df=0.9, method='we', labels_name='rating', text_column='review'):
    """
    :param input_path: path of the csv to read
    :param method: method for the embedding
    :param text_column: name of the column of the csv to use for the text items
    :param labels: name of the column to use for the labels, if none enter 0
    :return: train, test with labels
    """
    data = pd.read_csv(input_path, nrows=50)
    text = data[text_column].values

    # careful if no labels are provided
    if labels_name:
        labels = data[labels_name].values
    else:
        labels = np.zeros(text.shape)
    # split data
    raw_train_data, raw_test_data, train_labels, test_labels = train_test_split(text, labels,
                                                                                test_size=0.33, random_state=42)

    train_labels = train_labels[:, np.newaxis]
    test_labels = test_labels[:, np.newaxis]

    # Do the appropriate embedding on the text
    if method == 'we':
        train_data, test_data = we_features(raw_train_data, raw_test_data, langage, fname, b,
                                            seq_l, ngram=ngram, min_df=min_df, max_df=max_df)
    elif method == 'bow':
        train_data, test_data = bow_features(raw_train_data, raw_test_data, langage,
                                             ngram=ngram, min_df=min_df, max_df=max_df)
    else:
        raise ValueError('This is not an acceptable method !')

    # return the approriate data
    columns = ['text']
    if labels_name:
        return train_data, test_data, train_labels, test_labels
    #     columns.append('label')
    #     train = hstack((train_data, train_labels))
    #     test = hstack((test_data, test_labels))
    #     train_data = pd.DataFrame(data=train.toarray())
    #     test_data = pd.DataFrame(data=test.toarray())
    # else:
    #     train_data = pd.DataFrame(train_data, columns=columns)
    #     test_data = pd.DataFrame(test_data, columns=columns)
    # if save_name:
    #     train_data.to_csv('train_data/' + save_name)
    #     test_data.to_csv('train_data/' + save_name)
    return train_data, test_data


if __name__ == '__main__':
    pass
    a = create_features('train_data/raw_csv/imdb.csv', 'en', method='we', save_name='test.csv')
    print(a)
    # embedding_fname = 'train_data/word_embeddings/GoogleNews-vectors-negative300.bin'
    # binary = True
    # data = [["I", "eat", "a", "cow"],
    #         ["the", "bull", "is", "dead"]]
    # seq_len = 5
    #
    # X = word_embeddings(embedding_fname, binary, data, seq_len)
    # print(X[0, 4, 0:5])
    # print(X[0, 2, 0:5])
