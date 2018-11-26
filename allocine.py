# Import libs
import pandas as pd
import os
from requests import get
from time import sleep
from random import randint
from bs4 import BeautifulSoup
import bs4

from warnings import warn
from time import time


# Function to scrape the movies urls from http://www.allocine.fr/films/
# Choose the page range with the two parameters start_page and end_page.
# The url list is save as a csv file: movie_url.csv
def getMoviesUrl(start_page, end_page, path='data/allocine/url/'):
    # Set the list
    movies_list = []
    movies_number = 0
    for p in range(start_page, end_page + 1):
        # Get request
        url = 'http://www.allocine.fr/films/?page={}'.format(str(p))
        response = get(url)

        # Pause the loop
        sleep(randint(1, 2))

        # Warning for non-200 status codes
        if response.status_code != 200:
            warn('Page Request: {}; Status code: {}'.format(p, response.status_code))

        # Parse the content of the request with BeautifulSoup
        html_soup = BeautifulSoup(response.text, 'html.parser')

        # Select all the movies url from a single page
        movies = html_soup.find_all('h2', 'meta-title')
        movies_number += len(movies)

        for movie in movies:
            url = (movie.a['href'])
            id = url[url.find('=') + 1: -5]
            row = {'id': id, 'page': p}
            movies_list.append(row)
    # Saving the files
    data = pd.DataFrame(data=movies_list)
    path = path + '{}_{}.csv'.format(start_page, end_page)
    data.to_csv(path_or_buf=path)


# t1 = time()
# a = getMoviesUrl(1, 15)
# print(time() - t1)
# fetches 20 000 id/hour

def chunks_movie_url(start_page, end_page, chunksize=20, path='data/allocine/url/'):
    """
    Wrapping of the previous method in one that does chunks to avoid loosing info
    """
    indexes = [(i, i + chunksize - 1) for i in range(start_page, end_page, chunksize)]
    for start_page, end_page in indexes:
        getMoviesUrl(start_page, end_page, path=path)
        print('processed {} pages'.format(end_page))


# t1 = time()
# chunks_movie_url(1, 20)
# print(time() - t1)
# same rate


def get_reviews_press(url, threshold=10):
    """
    fetch the reviews for one url for allociné reviews page
    :param url: query
    :param threshold: number of comments to fetch for this movie
    :return: list of tuples (text, rating)
    """
    # access the page
    response = get(url)

    # check if fetched
    if response.status_code != 200:
        return []

    # go to the list of press review
    movie_html_soup = BeautifulSoup(response.text, 'html.parser')
    review_section = movie_html_soup.body.main.div.find(id='content-start')
    reviews_elements = review_section.find(class_='reviews-press-comment')

    fetched = 0
    result = []
    # for each review, fetch the text of the review and get the number of star
    for child in reviews_elements.children:
        # if we have enough from this movie
        if fetched >= threshold:
            break

        # check if it is a review, parse the content, put it in a tuple
        if isinstance(child, bs4.element.Tag):
            review = str(child.p.string)
            review_class = (child.div.div.div['class'])
            rating = int(review_class[1][1])
            result.append((review, rating))
            fetched += 1
    return result


# test = 'http://www.allocine.fr/film/fichefilm-228086/critiques/presse/'
# reviews = get_reviews_press(test)
# print(reviews)


def get_reviews_spectator(url, threshold=10):
    """
    fetch the reviews for one url for allociné reviews page
    :param url: query
    :param threshold: number of comments to fetch for this movie
    :return: list of tuples (text, rating)
    """
    # access the page
    response = get(url)

    # check if fetched
    if response.status_code != 200:
        warn('Request #{}; Status code: {}'.format(0, response.status_code))
        return []

    # go to the list of press review
    movie_html_soup = BeautifulSoup(response.text, 'html.parser')
    reviews_elements = movie_html_soup.find('div', class_='reviews-users-comment')

    fetched = 0
    result = []
    # for each review, fetch the text of the review and get the number of star
    for child in reviews_elements.children:
        # if we have enough from this movie
        if fetched >= threshold:
            break
        # check if it is a review, parse the content, put it in a tuple
        if isinstance(child, bs4.element.Tag):
            test = child.find(class_='content-txt')
            if test is None:
                continue
            review = str(test.string)
            review_rating = (child.find(class_='stareval-note').string.strip().replace(',', '.'))
            rating = float(review_rating)
            result.append((review, rating))
            fetched += 1
    return result


# test2 = 'http://www.allocine.fr/film/fichefilm-235582/critiques/spectateurs/'
# reviews = get_reviews_spectator(test2)
# print(reviews)


def process_movie_list(movie_list, page, output_path='data/allocine/data/', threshold=10):
    """
    Given a list of ids,
    :param movie_list: list of ids
    :param page: page from which the ids where fetched
    :param output_path: where to save the extracted data
    :return: (result df, error dataframe)
    """
    # init list to save errors
    errors = []
    data = []

    # request loop
    for id in movie_list:
        try:
            spectator_url = 'http://www.allocine.fr/film/fichefilm-{}/critiques/spectateurs/'.format(id)
            press_url = 'http://www.allocine.fr/film/fichefilm-{}/critiques/presse/'.format(id)

            items = get_reviews_spectator(spectator_url, threshold)
            items += get_reviews_press(press_url, threshold)
            if not items:
                continue
            for text, rating in items:
                row = {'id': id, 'review': text, 'rating': rating}
                data.append(row)
        except:
            errors.append(id)
    c = ['id', 'review', 'rating']
    df = pd.DataFrame(columns=c, data=data)
    df.to_csv(output_path + "movies" + str(page) + ".csv")
    errors = pd.DataFrame(errors)
    errors.to_csv(output_path + "errors.csv", mode='a')
    return df, errors


# Load the list of urls
# df, error = process_movie_list(['2352, 2485'])
# print(df)


def process_all_fetched(start_page, end_page, threshold=10):
    """
    Process all url fetched in the url/ directory that have pages in the given range
    """
    input_dir = 'data/allocine/url/'
    for csv_name in sorted(os.listdir(input_dir), key=lambda x: int(x[:x.find('_')])):
        # only fetch csv that have pages between the requested ones
        start_csv, end_csv = int(csv_name[0:csv_name.find('_')]), \
                             int(csv_name[csv_name.find('_') + 1:csv_name.find('.')])
        if end_csv >= start_page and start_csv <= end_page:
            # if some pages are interesting, read the csv and compute the relevant pages
            data = pd.read_csv(input_dir + csv_name, usecols=['id', 'page'])
            data_grouped = data.groupby('page')
            # process page
            for page, data in data_grouped:
                if start_page <= page <= end_page:
                    movie_list = data['id'].values
                    process_movie_list(movie_list, page, threshold=threshold)
            print('processed {}'.format(csv_name))


t1 = time()
process_all_fetched(928, 1002)
print(time() - t1)
# 12395s for 500 pages


# done 10 pages (approx 1500 reviews) in 222s

def get_data():
    """
    load all reviews in a big csv
    :return: Dataframe
    """
    input_dir = 'data/allocine/data/'
    data = []
    for csv in os.listdir(input_dir):
        if csv != 'errors.csv':
            df = pd.read_csv(input_dir + csv, usecols=['id', 'review', 'rating'])
            data.append(df)
    return pd.concat(data, axis=0)
# print(get_data())
