from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
import pandas as pd

stop_words_fr = get_stop_words('fr')
stop_words_en = get_stop_words('en')


def create_vectorizer(data, stopwords, ngram=(1, 1)):
    """
    fit an instance of a vectorizer on a certain data
    :param data:
    :param m_df:
    :param ngram:
    :param stopwords:
    :return:
    """
    vectorizer = CountVectorizer(input='content ', analyzer='word',
                                 ngram_range=ngram
                                 , stop_words=stop_words_fr)
    vectorizer.fit(data)
    return vectorizer


# vec = create_vectorizer(['\n    \n              La Chance de ma vie est'])
# print(vec.transform(['\n    \n              La Chance de ma vie est vie',
#                      '\n La Chance de ma vie est']).toarray())


# data = pd.read_csv('data/raw_csv/imdb.csv')
# reviews = data['review']
# vec = create_vectorizer(reviews, stop_words_fr)
# print(vec.transform(reviews))
