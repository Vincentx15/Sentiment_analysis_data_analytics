import wikipediaapi
import wikipedia
import requests
import pandas as pd

wiki_en = wikipediaapi.Wikipedia('en')
wiki_fr = wikipediaapi.Wikipedia('fr')


# page_py = wiki_en.page('Python_(programming_language)')
# page_fr = wiki_fr.page('Poulet')
# page_en = wiki_en.page('Chicken')


def get_other_language(page, language):
    try:
        langlinks = page.langlinks
        v = langlinks[language]
    except:
        # print('Pas de page en %s' % language)
        return 0
    return v


# params = {
#     'list': 'random',
#     'rnnamespace': 0,
#     'rnlimit': 10,
# }
# headers = {
#     'User-Agent': 'wikipedia (https://github.com/goldsmith/Wikipedia/)'
# }
# API_URL = 'http://en.wikipedia.org/w/api.php'
# r = requests.get(API_URL, params=params, headers=headers)
#
# r = requests.get(API_URL, params=params, headers=headers)
# print(r.text)
# r.json()

def sample_bilingual(number=10000):
    wikipedia.set_lang('fr')
    random_pages = (wikipedia.random(number))
    count = 0
    data = []
    for i, page_fr in enumerate(random_pages):
        if not i % 30:
            print('pages visited =', i)
        page_fr = wiki_fr.page(page_fr)
        page_en = get_other_language(page_fr, 'en')
        if page_en:
            count += 1
            id = page_fr.pageid
            summary_fr = page_fr.summary
            summary_en = page_en.summary
            row = {'id': id, 'summary_fr': summary_fr, 'summary_en': summary_en}
            data.append(row)
            if not count % 10:
                print('pages extracted =', count)
                df = pd.DataFrame(data=data)
                df.to_csv('train_data/wikipedia/samples.csv', mode='a')
                data = []
    df = pd.DataFrame(data=data)
    df.to_csv('train_data/wikipedia/samples.csv', mode='a')


sample_bilingual(10000)


# df = pd.read_csv('train_data/wikipedia/samples.csv')
# print(df)


def print_links(page):
    links = page.links
    for title in sorted(links.keys()):
        print("%s: %s" % (title, links[title]))

# print_links(page_py)


# def print_categories(page):
#     categories = page.categories
#     for title in sorted(categories.keys()):
#         print("%s: %s" % (title, categories[title]))
#
#
# print("Categories")
# print_categories(page_py)
#
#
# def print_categorymembers(categorymembers, level=0, max_level=2):
#     for c in categorymembers.values():
#         print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
#         if c.ns == wikipediaapi.Namespace.CATEGORY and level <= max_level:
#             print_categorymembers(c.categorymembers, level + 1)
#
#
# cat = wiki_en.page("Category:Physics")
# print("Category members: Category:Physics")
# # print_categorymembers(cat.categorymembers,max_level=0)
#
# dump=0
# wiki = gn.WikiCorpus(dump)
# for text in wiki.get_texts():
#     print(text)
