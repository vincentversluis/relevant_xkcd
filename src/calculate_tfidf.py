# %% HEADER
# Calculate TF-IDF vectors for each heading and store in database

# https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=Term%20Frequency%3A%20TF%20of%20a,of%20words%20in%20the%20document.&text=Inverse%20Document%20Frequency%3A%20IDF%20of,corpus%20that%20contain%20the%20term.

# TODO: Consider removing character names from strings

# %% IMPORTS
from nltk.corpus import stopwords
import re
import string
from tqdm import tqdm
from functools import cache
import pandas as pd
from utils import db_utils
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# %% INPUTS
db_path = '../data/relevant_xkcd.db'
n_gram_max_length = 3

# %% INITIALISINGS
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
lemmatizer = WordNetLemmatizer()

# Build split pattern with punctuation and word boundaries for stopwords
STOPWORD_PATTERN = r'\b(?:' + '|'.join(map(re.escape, stopwords.words('english'))) + r')\b'
PUNCTUATION_PATTERN = '|'.join(map(re.escape, string.punctuation))
SPLIT_PATTERN = f'({STOPWORD_PATTERN})|({PUNCTUATION_PATTERN})'

# %% GET DATA
xkcd_properties = db_utils.get_xkcd_properties(db_path)
xkcd_explanations = db_utils.get_xkcd_explained(db_path)

# %% FUNCTIONS
def get_up_to_ngrams(text: str, n: int=2, also_return_lower_case: bool=True) -> list:
    """Return a list of up-to-n-grams from the text. The returned list contains all
    separate single words in the text, but also n-grams of the text, with size up
    to n. (e.g. n=3 returns 1-grams, 2-grams, 3-grams). The n-grams of n>1 are composed
    of the words not separated by stopwords and punctuation. For example, the text
    "I have a cocktail, which is very nice" for n=2 would return all single words
    and the 2-grams I_have and very_nice, but not cocktail_which, assuming stopwords
    are "a" and "is".
    
    This works for English text.
    
    Args:
        text (str): The text to split into n-grams.
        n (int, optional): _description_. Defaults to 2.
    """
    # Split using regex (captures both stopwords and punctuation)
    fragments = re.split(SPLIT_PATTERN, text, flags=re.IGNORECASE)
    # Filter out empty strings and spaces
    fragments = [
        fragment 
        for fragment 
        in fragments 
        if fragment 
        and not re.fullmatch(SPLIT_PATTERN, fragment, flags=re.IGNORECASE)
        ]
    fragments = [fragment.strip().split() for fragment in fragments if fragment.strip()]
    
    # Collect all up to n grams
    up_to_ngrams = []
    for fragment in fragments:
        for i in range(1, n+1):
            up_to_ngrams += ['_'.join(fragment[j:j+i]) for j in range(len(fragment)+1-i)]
    
    # Optionally also return all lower case versions of the up to n grams
    if also_return_lower_case:
        up_to_ngrams += [word.lower() for word in up_to_ngrams]
    
    return up_to_ngrams

@cache
def get_wordnet_pos(tag):
    if tag.startswith('J'):  
        return 'a'
    elif tag.startswith('V'):  
        return 'v'
    elif tag.startswith('N'):  
        return 'n'
    elif tag.startswith('R'):  
        return 'r'
    else:
        return 'n' 
        
@cache
def lemmatise_sentence(sentence):
    # Prepare tokens and tagged tokens
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)

    # Lemmatise per token and collect sentence
    lemmatised_sentence = []
    for word, tag in tagged_tokens:
        if word.lower() == 'are' or word.lower() in ['is', 'am']:
            lemmatised_sentence.append(word)  
        else:
            lemmatised_sentence.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
    
    # Join up the sentence
    lemmatised_sentence = ' '.join(lemmatised_sentence)
    
    return lemmatised_sentence

sentence = "The children are running towards a better place."
sentence = "Creating I am, no he'll yet creating fifth be, creating a, cocktailer, [cocktailing] menus party oven stockracings for a PARTY poopers hangout in New York City, which has a party place where party people party"

# get_up_to_ngrams(sentence, n=3)
lemmatised_sentence = lemmatise_sentence(sentence)
get_up_to_ngrams(lemmatised_sentence, n=n_gram_max_length)

# %%
# Preprocess text
for xkcd in xkcd_properties:
    text = xkcd.title_text
    print(text)
    lemmatised_text = lemmatise_sentence(text)
    up_to_ngrams = get_up_to_ngrams(lemmatised_text, n=3, also_return_lower_case=True)
    for ngram in up_to_ngrams:
        print(ngram)
    print(50*'-')
    
    
# %%
