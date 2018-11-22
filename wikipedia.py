import wikipediaapi
import gensim.corpora as gn

wiki_en = wikipediaapi.Wikipedia('en')
wiki_fr = wikipediaapi.Wikipedia('fr')

page_py = wiki_en.page('Python_(programming_language)')
page_fr = wiki_fr.page('Poulet')
page_en = wiki_en.page('Chicken')


def get_other_language(page, language):
    langlinks = page.langlinks
    try:
        v = langlinks[language]
    except:
        print('Pas de page en %s'%language)
        return 0
    return v


page_fr_en = get_other_language(page_en, 'fr')
print(page_fr_en==page_fr)


def print_links(page):
    links = page.links
    for title in sorted(links.keys()):
        print("%s: %s" % (title, links[title]))


print_links(page_py)


def print_categories(page):
    categories = page.categories
    for title in sorted(categories.keys()):
        print("%s: %s" % (title, categories[title]))


print("Categories")
print_categories(page_py)


def print_categorymembers(categorymembers, level=0, max_level=2):
    for c in categorymembers.values():
        print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
        if c.ns == wikipediaapi.Namespace.CATEGORY and level <= max_level:
            print_categorymembers(c.categorymembers, level + 1)


cat = wiki_en.page("Category:Physics")
print("Category members: Category:Physics")
# print_categorymembers(cat.categorymembers,max_level=0)

dump=0
wiki = gn.WikiCorpus(dump)
for text in wiki.get_texts():
    print(text)
