# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:28:23 2018

@author: Clément Jumel
"""

import tweepy
import time
import datetime


def complete_filename(file, query, lang, max_tweets, extended):
    '''
    Complete the filename by adding the folder, the extansion, etc
    '''
    if extended:
        str_extended = '_extended'
    else:
        str_extended = ''
    if query == "*":
        return 'data/twitter2/' + file + '_' + lang + '_' + str(max_tweets) + '_' + 'all' + str_extended + '.txt'
    else:
        return 'data/twitter2/' + file + '_' + lang + '_' + str(max_tweets) + '_' + query + str_extended +'.txt'


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


def search_tweets(api, query, lang, max_tweets, min_age, extended):
    '''
    Search the tweets specified by the parameters
    '''
    if min_age == 0:
        # Search tweets according to query and lang only
        if extended:
            return [status for status in handle_error(
                tweepy.Cursor(api.search, q=query, lang=lang, tweet_mode='extended').items(max_tweets))]
        else:
            return [status for status in handle_error(tweepy.Cursor(api.search, q=query, lang=lang).items(max_tweets))]

    else:
        # Define the tweet date
        td = datetime.datetime.now() - datetime.timedelta(days=min_age - 1)
        tweet_date = '{0}-{1:0>2}-{2:0>2}'.format(td. year, td.month, td.day)

        # Search tweets according to query and lang
        if extended:
            return [status for status in
                handle_error(tweepy.Cursor(api.search, q=query, lang=lang, until=tweet_date, tweet_mode='extended').items(max_tweets))]
        else:
            return [status for status in
                handle_error(tweepy.Cursor(api.search, q=query, lang=lang, until=tweet_date).items(max_tweets))]


def write_csv(fname, tweets, extended):
    '''
    Write the tweets data in a csv
    '''
    # Open the file
    with open(fname, 'w', encoding="utf-8") as f:
        # Process tweet by tweet
        for tweet in tweets:
            if extended:
                if 'retweeted_status' in dir(tweet):
                    text = tweet.retweeted_status.full_text
                else:
                    text = tweet.full_text
            else:
                text = tweet._json['text']

            # Save it as a csv
            f.write(str(tweet._json['id']) + ', ' + text.replace("\n", "") + '\n')
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
    file = "twitter_server_2"  # No need to specify the folder or the file extension
    max_tweets = 2000  # Max number of tweets
    min_age = 0  # Min age of the tweets in days (0 means most recent tweets)
    extended = True

    # Load twitter's api
    api = load_api('data/twitter/twitter_keys.txt')

    i = 0
    while True:
        i += 1
        lang = "fr"
        query = "*"
        fname = complete_filename(file+'_'+str(i)+'_', query, lang, max_tweets, extended)
        tweets = search_tweets(api, query, lang, max_tweets, min_age, extended)
        write_csv(fname, tweets, extended)

        lang = "en"
        query = "*"
        fname = complete_filename(file+'_'+str(i)+'_', query, lang, max_tweets, extended)
        tweets = search_tweets(api, query, lang, max_tweets, min_age, extended)
        write_csv(fname, tweets, extended)

    '''
    # Search tweets specified by the query
    tweets = search_tweets(api, query, lang, max_tweets, min_age)

    # Write the tweets in a json file
    write_csv(fname, tweets)
    '''
