# %% HEADER
# Calculate TF-IDF vectors for each heading and store in database
# TODO: Consider removing character names from strings
# TODO: Rework comments to nicer language
# TODO: What if semantics return nothing? Then first try destopworded search in titles and such, then literal words
# TODO: Strip upper case using NER
# TODO: Docstrings
# TODO: Scraping seems to have gone a bit wrong on title text xkcd 3 - With original captions

# %% IMPORTS
from functools import cache
from itertools import chain
import regex as re

import sqlite3
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import db_utils

# %% INITIALISINGS
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
lemmatizer = WordNetLemmatizer()

# %% FUNCTIONS
def get_tfidf(
    df: pd.DataFrame,
    doc_id_col: str,
    text_col: str,
    split_pattern: str,
    n_gram_length: int = 1,
) -> pd.DataFrame:
    df["tokens"] = df[text_col].apply(
        lambda text: " ".join(
            get_preprocessed_tokens(
                text, 
                n_gram_length=n_gram_length, 
                split_pattern=split_pattern
            )
        )
    )
    df = df.groupby(doc_id_col)['tokens'].agg(' '.join).reset_index()
    
    # Fit TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=False, 
        tokenizer=lambda x: x.split(), 
        preprocessor=lambda x: x)
    tfidf_matrix = vectorizer.fit_transform(df["tokens"])

    # Get non-zero entries
    nonzero_indices = tfidf_matrix.nonzero()
    rows, cols = nonzero_indices
    scores = tfidf_matrix.data

    # Map back to tokens and doc_ids
    tokens = [vectorizer.get_feature_names_out()[i] for i in cols]
    doc_ids = df[doc_id_col].iloc[rows].values

    tfidf_df = pd.DataFrame({
        doc_id_col: doc_ids,
        "token": tokens,
        "tfidf": scores
    })
    return tfidf_df

def get_ngrams(text: str, n: int) -> list:
    """Return a list of n-grams from the text.

    This works for English text.

    Args:
        text (str): The text to split into n-grams.
        n (int, optional): _description_. Defaults to 2.
    """
    # Collect all up to n grams
    text_split = text.split()
    if n > len(text_split):
        return []
    else:
        ngrams = [
            "_".join(text_split[j : j + n]) for j in range(len(text_split) + 1 - n)
        ]
    return ngrams

@cache
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return "a"
    elif tag.startswith("V"):
        return "v"
    elif tag.startswith("N"):
        return "n"
    elif tag.startswith("R"):
        return "r"
    else:
        return "n"

def lemmatise_sentence(sentence):
    # Prepare tokens and tagged tokens
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)

    # Lemmatise per token and collect sentence
    lemmatised_sentence = []
    for word, tag in tagged_tokens:
        if word.lower() == "are" or word.lower() in ["is", "am"]:
            lemmatised_sentence.append(word)
        else:
            lemmatised_sentence.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

    # Join up the sentence
    lemmatised_sentence = " ".join(lemmatised_sentence)

    return lemmatised_sentence

@cache
def get_preprocessed_tokens(
    text: str, split_pattern: str, n_gram_length: int = 1
) -> list:
    text_lemmatised = lemmatise_sentence(text)
    # print(text_lemmatised)
    text_split = split_text_by_regex(text_lemmatised, split_pattern=split_pattern)
    # print(text_split)
    tokens = list(
        chain(*[
            get_ngrams(fragment, n=n_gram_length)
            for fragment 
            in text_split])
    )
    return tokens

def split_text_by_regex(text: str, split_pattern: str) -> list:
    # Protect hyphenated words
    text_protected = re.sub(r"(?<=\w)-(?=\w)", "HYPHEN", text)
    
    # Split using regex
    fragments = re.split(split_pattern, text_protected, flags=re.IGNORECASE)
    
    # Filter out empty strings and spaces
    fragments = [
        fragment
        for fragment in fragments
        if fragment and not re.fullmatch(split_pattern, fragment, flags=re.IGNORECASE)
    ]
    fragments = [fragment.strip() for fragment in fragments if fragment.strip()]
    
    # Put hyphens back in
    fragments = [fragment.replace("HYPHEN", "-") for fragment in fragments]
    
    return fragments

# db_path = "../data/relevant_xkcd.db"
# xkcd_explanations_df = db_utils.get_xkcd_explained(db_path)
# xkcd_explanations_df = xkcd_explanations_df[xkcd_explanations_df['xkcd_id'].isin([1])]

# n_gram_length = 1

# stopwords_pattern = (
#     r"\b(?:" + "|".join(map(re.escape, stopwords.words("english"))) + r")\b"
# )
# punctuation_pattern = r'[^\p{L}\p{M}\p{N} ]+'
# split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"

# # Transcripts
# import unicodedata
# df = xkcd_explanations_df[
#     xkcd_explanations_df["heading"] == "Transcript"
# ]
# # df["heading"] = df["heading"].apply(lambda x: unicodedata.normalize("NFC", x))
# text_col = "text"
# tfidf_df = get_tfidf(
#     df, 
#     doc_id_col="xkcd_id", 
#     text_col=text_col, 
#     n_gram_length=n_gram_length, 
#     split_pattern=split_pattern)
# tfidf_df['heading'] = "transcript"
# # tfidf_df = tfidf_df.drop_duplicates(subset=['xkcd_id', 'token', 'tfidf'], keep=False)
# print(f"\nFound {len(tfidf_df)} tf-idf scores for transcripts with n_gram_length {n_gram_length}")
# tfidf_df

# # %%
# df

# # %%
# tfidf_df[tfidf_df['token'] == 'barrel']
# # %%
