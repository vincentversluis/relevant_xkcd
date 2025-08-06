# %% HEADER
# Calculate TF-IDF vectors for each heading and store in database

# https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=Term%20Frequency%3A%20TF%20of%20a,of%20words%20in%20the%20document.&text=Inverse%20Document%20Frequency%3A%20IDF%20of,corpus%20that%20contain%20the%20term.

# TODO: Consider removing character names from strings
# TODO: Check fill rate for each explanation, title and title text
# TODO: Rework comments to nicer language
# TODO: What if semantics return nothing? Then first try destopworded search in titles and such, then literal words

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
from sklearn.feature_extraction.text import TfidfVectorizer

# %% INPUTS
db_path = '../data/relevant_xkcd.db'
n_gram_max_length = 3

# Which headings to ignore for tf-idf calculation
headings_ignore = ['Trivia']

# %% INITIALISINGS
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
lemmatizer = WordNetLemmatizer()

# Build split pattern with punctuation and word boundaries for stopwords
STOPWORD_PATTERN = r'\b(?:' + '|'.join(map(re.escape, stopwords.words('english'))) + r')\b'
PUNCTUATION_PATTERN = '|'.join(map(re.escape, string.punctuation))
SPLIT_PATTERN = f'({STOPWORD_PATTERN})|({PUNCTUATION_PATTERN})'

# %% GET DATA
xkcd_properties_df = db_utils.get_xkcd_properties(db_path)
xkcd_explanations_df = db_utils.get_xkcd_explained(db_path)

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
        up_to_ngrams += [word.lower() for word in up_to_ngrams if word.lower() != word]
    
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

@cache
def get_preprocessed_tokens(text: str, n_gram_max_length: int=1, also_return_lower_case: bool=True) -> list:
    lemmatised_text = lemmatise_sentence(text)
    up_to_ngrams = get_up_to_ngrams(
        lemmatised_text, 
        n=n_gram_max_length, 
        also_return_lower_case=also_return_lower_case)
    return up_to_ngrams

def get_tfidf(df: pd.DataFrame, doc_id_col: str, text_col: str) -> pd.DataFrame:
    # Preprocess
    df['terms'] = df[text_col].apply(
        lambda x: ' '.join(
            get_preprocessed_tokens(
            x, 
            n_gram_max_length=3, 
            also_return_lower_case=False)))  # TODO: Test with True / False
    
    # Fit TF-IDF
    vectorizer = TfidfVectorizer(lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(df['terms'])
    
    # Get feature names and document IDs
    terms = vectorizer.get_feature_names_out()
    doc_ids = df[doc_id_col].values
    
    # Create a DataFrame from the sparse matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=doc_ids, columns=terms)
    tfidf_df = tfidf_df.reset_index().rename(columns={'index': doc_id_col})

    # Melt the DataFrame to long format and only keep tf-idf scores
    title_text_tfidf_df = tfidf_df.melt(id_vars=doc_id_col, var_name='terms', value_name='tfidf')
    title_text_tfidf_df = title_text_tfidf_df[title_text_tfidf_df['tfidf'] > 0]
    return title_text_tfidf_df

# Get tfidf for title texts
get_tfidf(xkcd_properties_df, 'xkcd_id', 'title_text')

# %%
