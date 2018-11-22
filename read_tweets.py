# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:15:01 2018

@author: Clément Jumel
"""

import json
import pandas as pd

def read_json(fnames):
    '''
    Read the tweets in json files and return them as a list of dictionnaries 
    '''
    tweets = []
    for fname in fnames:
        # Open the file
        with open(fname) as f:
            # Read lines one by one
            for line in f:
                # Retrieve dictionnaries from json
                tweets.extend([json.loads(line)])
    return tweets

def generate_df(tweets,df_col,tweet_label):
    '''
    Create a dataframe from tweets with the designated columns
    '''
    df = pd.DataFrame()
    
    if len(df_col) != len(tweet_label):
        raise ValueError("DataFrame columns and Tweet labels haven't the same size")
    
    for i in range(len(df_col)):
        col,label = df_col[i],tweet_label[i]
        if len(label)==1:
            df[col] = [tweet[label[0]] for tweet in tweets]
        elif len(label)==2:
            df[col] = [tweet[label[0]][label[1]] for tweet in tweets]
    return df

if __name__ == "__main__":
    
    # Parameters
    fnames      = ['../data/twitter_en_10000_all_0.json'] # Names of the files to read
#                   '../data/twitter_fr_1000_macron_0.json']    
    df_col      = ['Date','Tweet']          # Name of the columns of the dataframe
    tweet_label = [['created_at'],['text']] # Name of the corresponding labels
    
    # Read the json files and create a list of tweet dictionnaries
    tweets = read_json(fnames)
    
    # Create a dataframe with the info of the tweets desinated by the parameters
    df = generate_df(tweets,df_col,tweet_label)