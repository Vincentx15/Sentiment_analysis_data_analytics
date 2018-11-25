from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
import pandas as pd
from sklearn.model_selection import train_test_split

stop_words_fr = get_stop_words('fr')
stop_words_en = get_stop_words('en')


def create_vectorizer(data, stopwords, ngram=(1, 1), min_df=0.01, max_df=0.9):
    """
    fit an instance of a vectorizer on a certain train_data
    :param data:
    :param m_df:
    :param ngram:
    :param stopwords:
    :return:
    """
    vectorizer = CountVectorizer(input='content ',
                                 analyzer='word',
                                 ngram_range=ngram,
                                 stop_words=stopwords,
                                 min_df=min_df,
                                 max_df=max_df)
    vectorizer.fit(data)
    return vectorizer


# vec = create_vectorizer(['\n    \n              La Chance de ma vie est'])
# print(vec.transform(['\n    \n              La Chance de ma vie est vie',
#                      '\n La Chance de ma vie est']).toarray())


from sklearn.ensemble import AdaBoostRegressor
from joblib import dump, load

data = pd.read_csv('train_data/raw_csv/imdb.csv')
reviews = data['review']
labels = data['rating']
# vec = create_vectorizer(reviews, stop_words_en)
# dump(vec, 'vec.joblib')
vec = load('vec.joblib')
res = vec.transform(reviews)
print(res.shape)

# model = AdaBoostRegressor(n_estimators=300)
# model.fit(res, labels)
# dump(model, 'test.joblib')
# clf = load('test.joblib')


'''
Classifier
'''
from classifier import create_classifier, predict_classifier, rmse

Classifier = "NN"
Measure = "RMSE"

Parameters = {'NN_input_dim': 1681,
              'NN_layers': [600, 100, 10, 1],
              'NN_activations': ['relu', 'relu', 'relu', 'relu'],
              'NN_loss': 'mean_squared_error',
              'NN_optimizer': 'sgd',
              'epochs': 10,
              'batch_size': 32}

clf = create_classifier(Classifier, res, labels, Parameters)
error = rmse(clf.predict(res), labels)
print('rmse = ', error)

test = ['terrible movie. it was really great, do not go and see it',
        'Amazing movie, I loved seeing it. It was very fun. The actors play very well']
test = vec.transform(test)
print(test)
labels = clf.predict(test)
print(labels)

from textblob import TextBlob

testimonial = TextBlob('terrible movie. it was really boring, do not go and see it')
labels = testimonial.sentiment
print(labels)

testimonial = TextBlob('Amazing movie, I loved seeing it. It was very fun. The actors play very well')
labels = testimonial.sentiment
print(labels)
