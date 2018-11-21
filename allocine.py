# Import libs
import pandas as pd
import numpy as np
from requests import get
from time import time
from time import sleep
from random import randint
from bs4 import BeautifulSoup
import bs4
import dateparser

from warnings import warn
from IPython.core.display import clear_output
import traceback


# Function to scrape the movies urls from http://www.allocine.fr/films/
# Choose the page range with the two parameters start_page and end_page.
# The url list is save as a csv file: movie_url.csv
def getMoviesUrl(start_page, end_page):
    # Set the list
    movie_id = []
    p_requests = start_page
    movies_number = 0
    for p in range(start_page, end_page + 1):
        # Get request
        url = 'http://www.allocine.fr/films/?page={}'.format(str(p))
        response = get(url)

        # Pause the loop
        sleep(randint(1, 2))

        # Warning for non-200 status codes
        if response.status_code != 200:
            warn('Page Request: {}; Status code: {}'.format(p_requests, response.status_code))

        # Parse the content of the request with BeautifulSoup
        html_soup = BeautifulSoup(response.text, 'html.parser')

        # Select all the movies url from a single page
        movies = html_soup.find_all('h2', 'meta-title')
        movies_number += len(movies)

        for movie in movies:
            url = (movie.a['href'])
            id = url[url.find('=') + 1: -5]
            movie_id.append(id)

    # Saving the files
    r = np.asarray(movie_id)
    np.savetxt("movie_url.csv", r, delimiter=",", fmt='%s')


# a = getMoviesUrl(0, 0)


#
# print(a)

# m_url = pd.read_csv("movie_url.csv")


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
            if test == None:
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


def process_movie_list(movie_list):
    # init list to save errors
    errors = []
    data = []

    # request loop
    for id in movie_list:
        try:
            spectator_url = 'http://www.allocine.fr/film/fichefilm-{}/critiques/spectateurs/'.format(id)
            press_url = 'http://www.allocine.fr/film/fichefilm-{}/critiques/presse/'.format(id)

            items = get_reviews_spectator(spectator_url)
            items += get_reviews_press(press_url)
            if not items:
                continue
            for text, rating in items:
                row = {'id': id, 'review': text, 'rating': rating}
                data.append(row)
        except:
            errors.append(id)
    c = ['id', 'review', 'rating']
    df = pd.DataFrame(columns=c, data=data)
    df.to_csv("allocine_movies.csv")
    errors = pd.DataFrame(errors)
    errors.to_csv("allocine_errors.csv")
    return df, errors

# Load the list of urls
# m_url = pd.read_csv("movie_url.csv")
# df, error = process_movie_list(['2352'])
# print(df)
