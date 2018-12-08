# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:28:23 2018

@author: Clément Jumel
"""

import tweepy
import time
import datetime


def complete_filename(file, query, lang, max_tweets):
    '''
    Complete the filename by adding the folder, the extansion, etc
    '''
    if query == "*":
        return 'data/twitter/' + file + '_' + lang + '_' + str(max_tweets) + '_' + 'all' + '.txt'
    else:
        return 'data/twitter/' + file + '_' + lang + '_' + str(max_tweets) + '_' + query + '.txt'


def load_api(fname):
    '''
    Read the keys from a txt file and load twitter's api
    '''
    # Read the keys
    keys = []
    with open(fname) as f:
        # Read keys one by one
        for line in f:
            keys.extend([line.split('\n')[0]])
    [consumer_key, consumer_secret, access_token, access_secret] = keys

    # Authentication in Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    # Load and return Twitter's api
    return tweepy.API(auth)


def search_tweets(api, query, lang, max_tweets, min_age):
    '''
    Search the tweets specified by the parameters
    '''
    if min_age == 0:
        # Search tweets according to query and lang only
        # return [status for status in handle_error(tweepy.Cursor(api.search, q=query, lang=lang).items(max_tweets))]
        return [status for status in handle_error(tweepy.Cursor(api.search, q=query, lang=lang).items(max_tweets))]

    else:
        # Define the tweet date
        td = datetime.datetime.now() - datetime.timedelta(days=min_age - 1)
        tweet_date = '{0}-{1:0>2}-{2:0>2}'.format(td. year, td.month, td.day)

        # Search tweets according to query and lang
        return [status for status in
                handle_error(tweepy.Cursor(api.search, q=query, lang=lang, until=tweet_date).items(max_tweets))]


def write_csv(fname, tweets):
    '''
    Write the tweets data in a csv
    '''
    # Open the file
    with open(fname, 'w', encoding="utf-8") as f:
        # Process tweet by tweet
        for tweet in tweets:
            # Save it as a csv
            f.write(str(tweet._json['id']) + ', ' + tweet._json['text'].replace("\n", "") + '\n')
    return


def handle_error(Cursor):
    '''
    Handle the errors in the Cursor and wait in this case
    '''
    while True:
        try:
            yield Cursor.next()
        except tweepy.TweepError:
            print("Rate limit reached, sleeping for 15 minutes...")
            for t in range(3):
                time.sleep(5 * 60)
                print("{}/15...".format((t + 1) * 5))





if __name__ == '__main__':
    pass
    # parameters
    file = "twitter_server"  # No need to specify the folder or the file extension
    query = "*"  # Query ("*" means everything)
    lang = "fr"  # Language in the query
    max_tweets = 1  # Max number of tweets
    min_age = 0  # Min age of the tweets in days (0 means most recent tweets)


    # Load twitter's api
    api = load_api('data/twitter/twitter_keys.txt')

    i = 0
    while i < 10:
        i += 1
        lang = "fr"
        fname = complete_filename(file+'_'+str(i)+'_', query, lang, max_tweets)
        tweets = search_tweets(api, query, lang, max_tweets, min_age)
        write_csv(fname, tweets)

        lang = "en"
        fname = complete_filename(file+'_'+str(i)+'_', query, lang, max_tweets)
        tweets = search_tweets(api, query, lang, max_tweets, min_age)
        write_csv(fname, tweets)

    '''
    # Search tweets specified by the query
    tweets = search_tweets(api, query, lang, max_tweets, min_age)

    # Write the tweets in a json file
    write_csv(fname, tweets)
    '''
