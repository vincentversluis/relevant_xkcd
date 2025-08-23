# %% NOTES
# Preprocessing and tf-idf stuff
# https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/
# https://towardsdatascience.com/pipeline-for-text-data-pre-processing-a9887b4e2db3/
# https://medium.com/@devangchavan0204/complete-guide-to-text-preprocessing-in-nlp-b4092c104d3e

# Pretrained word embeddings
# https://medium.com/@gauravtailor43/googles-trained-word2vec-model-in-python-ab9b2df09af2

# %% HEADER
# Just prutsing
# TODO: Check what happens with punctuation removal and apostrophe's like in Huub's
# TODO: Docstrings
# TODO: Import order
# TODO: Calculate tfidf of inputs as well

# %% IMPORTS
import gensim.downloader as api
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import re
import string
from itertools import pairwise
from time import sleep
import arrow
from functools import cache
import sqlite3
import pandas as pd
from typing import Optional
from xkcd_tfidfing import get_preprocessed_tokens
import inflect


# %% INPUTS
db_path = "../data/relevant_xkcd.db"

n_gram_max_length = 3

# n_gram_weights = {1: 1, 2: 1.5, 3: 2}
n_gram_weights = 'length'

# # Build split pattern with punctuation and word boundaries for stopwords
stopwords_pattern = (
    r"\b(?:" + "|".join(map(re.escape, stopwords.words("english"))) + r")\b"
)
punctuation_pattern = r"[^\p{L}\p{M}\p{N} ]+"
split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"

# %% LOAD THINGS
# This is large (~1.5GB file), so download it once and use the cached version
# Loading it also takes a minute or so
word2vec = api.load("word2vec-google-news-300")
inflect_engine = inflect.engine()

# %% FUNCTIONS
@cache
def get_similar_words(word, topn=10):
    try:
        similar_words = word2vec.most_similar(word, topn=topn)
    except KeyError:  # Word is not in the model
        similar_words = []        
    return similar_words
        
def get_similar_words_for_list(words, topn=10):
    similar_words = []
    for word in words:
        similar_words.append({word: get_similar_words(word, topn=topn)})
    return similar_words

def get_xkcd_tfidfs(db_path, tokens):
    conn = sqlite3.connect(db_path)

    # Use parameterized query with IN clause
    placeholders = ','.join('?' for _ in tokens)
    query = f"""
        SELECT 
            xkcd_id
        ,   heading
        ,   token
        ,   tfidf
        FROM XKCD_EXPLAINED_TFIDF 
        WHERE token IN ({placeholders})
        """

    # Read directly into a DataFrame
    df = pd.read_sql_query(query, conn, params=tokens)

    # Close connection
    conn.close()

    return df

def get_xkcd_properties_for_ids(db_path, xkcd_ids):
    conn = sqlite3.connect(db_path)
    placeholders = ','.join('?' for _ in xkcd_ids)
    query = f"""
        SELECT 
            xkcd_id
        ,   title
        ,   date
        ,   title_text
        FROM XKCD_PROPERTIES
        WHERE xkcd_id IN ({placeholders})
        """

    # Read directly into a DataFrame
    df = pd.read_sql_query(query, conn, params=xkcd_ids)

    # Close connection
    conn.close()
    return df

def get_up_to_n_grams(text, split_pattern, n_gram_max_length=1):
    tokens = []
    for n_gram_length in range(1, n_gram_max_length + 1):
        tokens.extend(
            get_preprocessed_tokens(
                text, n_gram_length=n_gram_length, split_pattern=split_pattern
            )
        )
    return tokens

def get_ngram_weight(
    n_gram: str,
    n_gram_weights: dict | str | None = None,
    ) -> float:
    if n_gram_weights is None:
        return 1
    elif n_gram_weights == 'length':
        return len(n_gram.split('_'))
    elif isinstance(n_gram_weights, dict):
        try:
            return n_gram_weights[len(n_gram.split('_'))]
        except KeyError as e:
            raise ValueError(f"Did not pass a weight for ngram length {len(n_gram.split("_"))}") from e
    else:
        raise ValueError(f"Did not pass a valid ngram weighting method for {n_gram}")

def sub_number_to_words(text, numerical_threshold=20):
    num_words = {
        str(i): inflect_engine.number_to_words(i) 
        for i 
        in range(1, numerical_threshold + 1)}

    # Pattern matches whole numbers between 0 and 20 as standalone words
    pattern = re.compile(r'\b(?:' + '|'.join(num_words.keys()) + r')\b')

    return pattern.sub(lambda m: num_words[m.group()], text)

def recommend_xkcd(
    text: str, 
    top_n: int = 10,
    n_gram_max_length: int = 3, 
    n_gram_weights: dict | None = 'length',
    split_pattern: str | None = None,
    heading_weights: dict | None = None,
    number_to_words: bool = True,
    number_to_words_threshold: int = 20,
    ) -> pd.DataFrame:

    if split_pattern is None:
        stopwords_pattern = (
            r"\b(?:" + "|".join(map(re.escape, stopwords.words("english"))) + r")\b"
        )
        punctuation_pattern = r"[^\p{L}\p{M}\p{N} ]+"
        split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"
        
    # Create heading weight df
    if heading_weights is None:
        heading_weights = {
            'title': 1,  # Usually a description in only a few words
            'title_text': .1,  # Often this does not really pertain to the rest
            'transcript': 1,  # What people actually read
            'explanation': .5,  # Contains things that touch on the topic
        }
    heading_weights_df = pd.DataFrame.from_dict(heading_weights, orient='index').reset_index()
    heading_weights_df.columns = ['heading', 'weight']

    # Turn numerical words into written out numbers
    if number_to_words:
        text = sub_number_to_words(text, numerical_threshold=number_to_words_threshold)

    # Get all up to n-grams from text
    tokens = set(get_up_to_n_grams(
        text, 
        split_pattern=split_pattern, 
        n_gram_max_length=n_gram_max_length))

    similar_words = get_similar_words_for_list(tokens)

    # Create a df with the tokens and similar words
    similarwords_df = pd.DataFrame()
    for similar_word in similar_words:
        # Get current token
        token = list(similar_word.keys())[0]
        
        # Add entry with similarity of 1 to df
        similarwords_df = pd.concat([similarwords_df, pd.DataFrame([[token, float(1)]], columns=['token', 'similarity'])])
        
        # Get similar words from dict and add if not empty
        similarities = pd.DataFrame(similar_word[token], columns=['token', 'similarity'])
        if len(similarities) > 0:
            similarwords_df = pd.concat([similarwords_df, similarities])

    # Add n-gram weight
    similarwords_df['n_gram_weight'] = similarwords_df['token'].apply(lambda x: get_ngram_weight(x, n_gram_weights=n_gram_weights))

    # Get tf-idf scores for all similar words
    similar_words = similarwords_df['token'].unique()
    xkcd_tfidfs = get_xkcd_tfidfs(db_path, similar_words)

    # Combine dfs and calculate total token_score per xkcd_id and sort by score
    df = pd.merge(similarwords_df, xkcd_tfidfs, on='token', how='left')
    df = pd.merge(df, heading_weights_df, on='heading', how='left')
    df['token_score'] = df['tfidf'] * df['similarity'] * df['n_gram_weight'] * df['weight']
    df = df.groupby('xkcd_id')['token_score'].sum().sort_values(ascending=False)
    
    # Get top results and make presentable
    df = df.head(top_n).reset_index().rename(columns={'token_score': 'score'})
    df['score'] = df['score'].round(2)
    
    # Add some properties
    xkcd_properties_df = get_xkcd_properties_for_ids(db_path, df['xkcd_id'].unique())
    xkcd_properties_df['link'] = xkcd_properties_df['xkcd_id'].apply(lambda x: f"https://www.explainxkcd.com/wiki/index.php/{x}")
    xkcd_properties_df['id_title'] = xkcd_properties_df['xkcd_id'].astype(str) + ' - ' + xkcd_properties_df['title']
    df = pd.merge(df, xkcd_properties_df, on='xkcd_id', how='left')
    
    return df[['score', 'id_title', 'link']]

text = "I am Creating a cocktail \n menu party stockrace for a PARTY pooper in New York City, which has a party place where party people party"
# text = """
#     I want to create an algorithm that uses a music genre as an input, 
#     then gives me a list of bands that I expect will be announcing a 
#     tour in around half a year
#     """
# text = "Helping out a 20 69 melodic death metal band that wants to grow to 5 or 21"
# text = "Music DRM"
text = "Having a day with my work team to figure out what we find important in our work and what we can do to make sure we do the right thing"
text = "I am on an awayday with my team and we want to figure out what our mission statement is"
text = "I am creating a cocktail menu for people who are into whiskey"

df = recommend_xkcd(text, top_n=5)

for recommendation in df.itertuples():
    print(f"Score: {recommendation.score:.2f} - {recommendation.id_title}")
    print(recommendation.link)
    print()

# %%
