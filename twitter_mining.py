# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:28:23 2018

@author: Clément Jumel
"""

import tweepy
import json
import datetime
import time

def complete_filename(file,query,lang,max_tweets,min_age):
    '''
    Complete the filename by adding the folder, the extansion, etc
    '''
    if query == "*":
        return '../data/'+file+'_'+lang+'_'+str(max_tweets)+'_'+'all'+'_'+str(min_age)+'.json'
    else:
        return '../data/'+file+'_'+lang+'_'+str(max_tweets)+'_'+query+'_'+str(min_age)+'.json'

def load_api(consumer_key,consumer_secret,access_token,access_secret):
    '''
    Load twitter's api
    '''
    # Authentification in Twitter
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    # Load and return Twitter's api
    return tweepy.API(auth)

def search_tweets(api,query,lang,max_tweets,min_age):
    '''
    Search the tweets specified by the parameters
    '''
    if min_age == 0:
        # Search tweets according to query and lang only
        return [status for status in handle_error(tweepy.Cursor(api.search,q=query,lang=lang).items(max_tweets))]
    
    else:
        # Define the tweet date
        td = datetime.datetime.now()-datetime.timedelta(days=min_age-1)
        tweet_date = '{0}-{1:0>2}-{2:0>2}'.format(td.year,td.month,td.day)
        
        # Search tweets according to query and lang
        return [status for status in handle_error(tweepy.Cursor(api.search,q=query,lang=lang,until=tweet_date).items(max_tweets))]

def write_json(fname,tweets):
    '''
    Write the tweets data in a json file
    '''
    # Open the file
    with open(fname,'w') as f:
        # Process tweet by tweet
        for tweet in tweets:
            # Save it in a json
            f.write(json.dumps(tweet._json)+'\n')
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
                time.sleep(5*60)
                print("{}/15...".format((t+1)*5))
    return


if __name__ == "__main__":
    
    # Parameters
    file        = "twitter"     # No need to specify the folder or the file extension
    query       = "*"           # Query ("*" means everything)
    lang        = "fr"          # Language in the query
    max_tweets  = 10000         # Max number of tweets
    min_age     = 0             # Min age of the tweets in days (0 means most recent tweets)
    
    # Keys of Twitter's app
    consumer_key    = "b2gx16ZXtaO2ty53ThKz2nbBG"
    consumer_secret = "eQC7d2z709w6yKUUifMUzG275TixLKce9i7fYki27wEx2fIlGx"
    access_token    = "1062483187719421952-jvkpMyAfXQLlJXJGRDak5Ec67g27nP"
    access_secret   = "57G5UzqYjggE7AoMuU4xt1nyHpUsYkgvQd65jCB2QdfNk"
    
    # Complete the filename
    fname = complete_filename(file,query,lang,max_tweets,min_age)
    
    # Load twitter's api
    api = load_api(consumer_key,consumer_secret,access_token,access_secret)
    
    # Seach tweets specified by the query
    tweets = search_tweets(api,query,lang,max_tweets,min_age)
    
    # Write the tweets in a json file
    write_json(fname,tweets)