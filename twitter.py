import tweepy
import time
import datetime


def complete_filename(file, query, language, max_tweets, extended):
    """
    Complete the filename by adding the folder, the extansion, etc
    :param file: name of the file
    :param query: query performed
    :param language: language
    :param max_tweets: max number of tweets
    :param extended: if extended version of tweets
    :return: string with the complete file name to save
    """
    if extended:
        str_extended = '_extended'
    else:
        str_extended = ''
    if query == "*":
        return 'data/twitter2/' + file + '_' + language + '_' + str(max_tweets) + '_' + 'all' + str_extended + '.txt'
    else:
        return 'data/twitter2/' + file + '_' + language + '_' + str(max_tweets) + '_' + query + str_extended + '.txt'


def load_api(fname):
    """
    Load Twitter's api
    :param fname: file name of the keys
    :return: loaded api
    """
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


def search_tweets(api, query, language, max_tweets, min_age, extended):
    """
    Search the tweets specified by the parameters (wait if the maximal rate is reached)
    :param api: loaded Twitter api
    :param query: Twitter query to perform
    :param language: language of the query
    :param max_tweets: max number of tweets returned
    :param min_age: minimal age in days of the searched tweets
    :param extended: whether or not to search for extended tweets
    :return: list of tweets indexes and (extended) texts
    """
    if min_age == 0:
        # Search tweets according to query and language only
        if extended:
            return [status for status in
                    handle_error(tweepy.Cursor(api.search, q=query, lang=language, tweet_mode='extended')
                                 .items(max_tweets))]
        else:
            return [status for status in
                    handle_error(tweepy.Cursor(api.search, q=query, lang=language)
                                 .items(max_tweets))]

    else:
        # Define the tweet date
        td = datetime.datetime.now() - datetime.timedelta(days=min_age - 1)
        tweet_date = '{0}-{1:0>2}-{2:0>2}'.format(td.year, td.month, td.day)

        # Search tweets according to query and language
        if extended:
            return [status for status in
                    handle_error(tweepy.Cursor(api.search, q=query, lang=language, until=tweet_date,
                                               tweet_mode='extended')
                                 .items(max_tweets))]
        else:
            return [status for status in
                    handle_error(tweepy.Cursor(api.search, q=query, lang=language, until=tweet_date)
                                 .items(max_tweets))]


def write_csv(fname, tweets, extended):
    """
    Write the tweets data in a csv
    :param fname: name where to save
    :param tweets: tweets to save
    :param extended: whether or not the tweets are extended
    :return: /
    """
    # Open the file
    with open(fname, 'w', encoding="utf-8") as f:
        # Process tweet by tweet
        for tweet in tweets:
            if extended:
                # Case of retweets
                if 'retweeted_status' in dir(tweet):
                    text = tweet.retweeted_status.full_text
                else:
                    text = tweet.full_text
            else:
                text = tweet._json['text']

            # Save it as a csv
            f.write(str(tweet._json['id']) + ', ' + text.replace("\n", "") + '\n')
    return


def handle_error(cursor):
    """
    Handle the errors in the cursor and wait in this case
    :param cursor:
    :return:
    """
    while True:
        try:
            yield cursor.next()
        except tweepy.TweepError:
            print("Rate limit reached, sleeping for 15 minutes...")
            for t in range(3):
                time.sleep(5 * 60)
                print("{}/15...".format((t + 1) * 5))


if __name__ == '__main__':

    # parameters
    file = "twitter_server_2"  # No need to specify the folder or the file extension
    max_tweets = 2000  # Max number of tweets
    min_age = 0  # Min age of the tweets in days (0 means most recent tweets)
    extended = True  # Whether or not to load extended tweets

    # Load twitter's api
    api = load_api('data/twitter/twitter_keys.txt')

    # Infinite loop, made to run on a remote server and load tweets in several .txt files
    cmpt = 0
    while True:
        cmpt += 1
        language = "fr"
        query = "*"
        fname = complete_filename(file + '_' + str(cmpt) + '_', query, language, max_tweets, extended)
        tweets = search_tweets(api, query, language, max_tweets, min_age, extended)
        write_csv(fname, tweets, extended)

        language = "en"
        query = "*"
        fname = complete_filename(file + '_' + str(cmpt) + '_', query, language, max_tweets, extended)
        tweets = search_tweets(api, query, language, max_tweets, min_age, extended)
        write_csv(fname, tweets, extended)
