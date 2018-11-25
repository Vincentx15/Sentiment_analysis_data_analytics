import pandas as pd
import os
import time

paths = ['train_data/aclImdb/test/neg']
paths.append('train_data/aclImdb/test/pos')
paths.append('train_data/aclImdb/train/neg')
paths.append('train_data/aclImdb/train/pos')


def parse_imdb_reviews(path_dir):
    """
    parse one folder of imdb large review db
    :param path:
    :return:
    """
    reviews = []
    max_i = 0
    for review in os.listdir(path_dir):
        id, rating = int(review[0:review.find('_')]), int(review[review.find('_') + 1:review.find('.')])
        with open(os.path.join(path_dir, review)) as f:
            review_text = f.readline()
        max_i += 1
        row = {'id': id, 'rating': rating, 'review': review_text}
        reviews.append(row)

        # print(row)
        if not max_i % 500:
            print(max_i)
        # For testing
        # if max_i > 500:
        #     break
    return pd.DataFrame(reviews)


# t1 = time.time()
# parse_imdb_reviews(paths[0])
# print(time.time() - t1)


def get_data(paths=paths):
    """
    Parse all folders in an iterable 'paths'
    :return: Dataframe
    """
    dfs = []
    for path in paths:
        dfs.append(parse_imdb_reviews(path))
    return pd.concat(dfs, axis=0)

# t1 = time.time()
# df = get_data()
# print(df)
# print(df.columns)
# print(time.time() - t1)

# df.to_csv('train_data/raw_csv/imdb.csv')



