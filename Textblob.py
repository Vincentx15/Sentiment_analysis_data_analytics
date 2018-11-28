from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
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

import time
import pickle

subset = -1

'''
en
'''

# reviews_en = data_en['review'][:subset]
# results = [TextBlob(review).sentiment[0] for review in reviews_en]
# results = [2.5 * (res + 1) for res in results]
# pickle.dump(results, open('textblop.p', "wb"))

# results = pickle.load(open("textblop.p", "rb"))

# results = np.array(results)
# sns.distplot(results)
# plt.show()
#
# y = np.array(ratings_en[:subset])
# y = np.divide(y, 2)
# sns.distplot(y)
# plt.show()
#
# from sklearn.metrics import accuracy_score, mean_squared_error
#
# error = mean_squared_error(results, y)
# print(np.sqrt(error))

'''
fr
'''

subset = -1


reviews_fr = data_fr['review'][:subset]
results = [TextBlob(review, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[0] for review in reviews_fr]
results = [2.5 * (res + 1) for res in results]
pickle.dump(results, open('textblop-fr.p', "wb"))

# results = pickle.load(open("textblop.p", "rb"))

results = np.array(results)
sns.distplot(results)

y = np.array(ratings_fr[:subset])
sns.distplot(y)
plt.show()

from sklearn.metrics import accuracy_score, mean_squared_error

error = mean_squared_error(results, y)
print(np.sqrt(error))