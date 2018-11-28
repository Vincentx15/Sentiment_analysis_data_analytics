from textblob import TextBlob
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data_en = pd.read_csv('data/raw_csv/imdb.csv')
reviews_en = data_en['review']
ratings_en = data_en['rating']

# x = np.array([len(review) for review in reviews_en])
# sns.distplot(x)
# plt.show()
# x = np.array([len(review.split()) for review in reviews_en])
# sns.distplot(x)
# plt.show()

data_fr = pd.read_csv('data/raw_csv/allocine.csv')
reviews_fr = data_fr['review']
ratings_fr = data_fr['rating']

# x = np.array([len(review) for review in reviews_fr])
# sns.distplot(x)
# plt.show()
# x = np.array([len(review.split()) for review in reviews_fr if len(review.split())<500])
# sns.distplot(x)
# plt.show()

# import time
# import pickle

# subset = -1

# reviews_en = data_en['review'][:subset]
results_en = [TextBlob(review).sentiment[0] for review in reviews_en]
results_en = [2.5 * (res + 1) for res in results_en]
results_fr = [TextBlob(review).sentiment[0] for review in reviews_fr]
results_fr = [2.5 * (res + 1) for res in results_fr]
# pickle.dump(results, open('textblop.p', "wb"))

# results = pickle.load(open("textblop.p", "rb"))

# results_en = np.array(results_en)
# results_fr = np.array(results_fr)
# sns.distplot(results_en)
# sns.distplot(results_fr)
# plt.show()

# y = np.array(ratings_en[:subset])
# y = np.divide(y, 2)
# sns.distplot(y)
# plt.show()

# from sklearn.metrics import accuracy_score, mean_squared_error

# error = mean_squared_error(results, y)
# print(np.sqrt(error))

# testimonial = TextBlob('terrible movie. it was really boring, do not go and see it')
# labels = testimonial.sentiment
# print(labels,labels[0])
#
# testimonial = TextBlob('Amazing movie, I loved seeing it. It was very fun. The actors play very well')
# labels = testimonial.sentiment
# print(labels)
