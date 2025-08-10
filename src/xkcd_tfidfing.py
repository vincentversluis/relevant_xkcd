# %% HEADER
# Calculate TF-IDF vectors for each heading and store in database

# https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=Term%20Frequency%3A%20TF%20of%20a,of%20words%20in%20the%20document.&text=Inverse%20Document%20Frequency%3A%20IDF%20of,corpus%20that%20contain%20the%20term.

# TODO: Consider removing character names from strings
# TODO: Rework comments to nicer language
# TODO: What if semantics return nothing? Then first try destopworded search in titles and such, then literal words
# TODO: Consider a separate calculation for each length of ngram
# TODO: Strip upper case using NER
# TODO: Build in extra step of removing punctuation and stopwords
# TODO: Docstrings

# %% IMPORTS
from functools import cache
from itertools import chain
import re
import string

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from utils import db_utils

# %% INPUTS
db_path = "../data/relevant_xkcd.db"
n_gram_max_length = 3

# Which headings to ignore for tf-idf calculation
headings_ignore = ["Trivia"]

# %% INITIALISINGS
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
lemmatizer = WordNetLemmatizer()

# Build split pattern with punctuation and word boundaries for stopwords
stopwords_pattern = (
    r"\b(?:" + "|".join(map(re.escape, stopwords.words("english"))) + r")\b"
)
punctuation_pattern = "|".join(map(re.escape, string.punctuation))
split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"


# %% FUNCTIONS
def split_text_by_regex(text: str, split_pattern: str) -> list:
    # Split using regex (captures both stopwords and punctuation)
    fragments = re.split(split_pattern, text, flags=re.IGNORECASE)
    # Filter out empty strings and spaces
    fragments = [
        fragment
        for fragment in fragments
        if fragment and not re.fullmatch(split_pattern, fragment, flags=re.IGNORECASE)
    ]
    fragments = [fragment.strip() for fragment in fragments if fragment.strip()]
    return fragments


@cache
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
    text: str, n_gram_length: int = 1, split_pattern: str = split_pattern
) -> list:
    text_lemmatised = lemmatise_sentence(text)
    text_split = split_text_by_regex(text_lemmatised, split_pattern=split_pattern)
    tokens = list(
        chain(*[get_ngrams(fragment, n=n_gram_length) for fragment in text_split])
    )
    return tokens


def get_tfidf(
    df: pd.DataFrame,
    doc_id_col: str,
    text_col: str,
    n_gram_length: int = 1,
    split_pattern: str = split_pattern,
) -> pd.DataFrame:
    # Preprocess
    df["tokens"] = df[text_col].apply(
        lambda text: " ".join(
            get_preprocessed_tokens(
                text, n_gram_length=n_gram_length, split_pattern=split_pattern
            )
        )
    )

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(df["tokens"])

    # Create a DataFrame from the sparse matrix
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()
    )
    tfidf_df["xkcd_id"] = df["xkcd_id"]

    # Melt the DataFrame to long format and only keep tf-idf scores
    tfidf_df = tfidf_df.melt(id_vars=doc_id_col, var_name="tokens", value_name="tfidf")
    tfidf_df = tfidf_df[tfidf_df["tfidf"] > 0]

    # Rename columns
    tfidf_df = tfidf_df.rename(columns={"tokens": "token"})
    return tfidf_df


# %% GET DATA
xkcd_properties_df = db_utils.get_xkcd_properties(db_path)
xkcd_explanations_df = db_utils.get_xkcd_explained(db_path)

# %% ANALYSE DATA
for n_gram_length in tqdm(
    range(1, n_gram_max_length + 1), desc="Calculating tf-idf scores..."
):
    # Get tfidf for titles
    tfidf_title_df = get_tfidf(
        xkcd_properties_df, "xkcd_id", "title", n_gram_length=n_gram_length
    )
    tfidf_title_df["heading"] = "title"

    # Get tfidf for title texts
    tfidf_title_text_df = get_tfidf(
        xkcd_properties_df, "xkcd_id", "title_text", n_gram_length=n_gram_length
    )
    tfidf_title_text_df["heading"] = "title_text"

    # Get tfidf for transcripts
    xkcd_transcripts_df = xkcd_explanations_df[
        xkcd_explanations_df["heading"] == "Transcript"
    ]
    xkcd_transcripts_df = xkcd_transcripts_df.groupby(
        ["xkcd_id", "heading"], as_index=False
    ).agg({"text": lambda x: " ".join(x)})
    tfidf_transcripts_df = get_tfidf(
        xkcd_transcripts_df, "xkcd_id", "text", n_gram_length=n_gram_length
    )
    tfidf_transcripts_df["heading"] = "transcript"

    # Get tfidf for rest of explanation, minus headings to ignore
    xkcd_rest_of_explanations_df = xkcd_explanations_df[
        ~xkcd_explanations_df["heading"].isin([headings_ignore, "Transcript"])
    ]
    xkcd_rest_of_explanations_df = xkcd_rest_of_explanations_df.groupby(
        ["xkcd_id"], as_index=False
    ).agg({"text": lambda x: " ".join(x)})
    tfidf_rest_of_explanations_df = get_tfidf(
        xkcd_rest_of_explanations_df, "xkcd_id", "text", n_gram_length=n_gram_length
    )

    # Report on tfidfs
    print(f"For n_gram_length = {n_gram_length}:")
    print(f"{len(tfidf_title_df)=}")
    print(f"{len(tfidf_title_text_df)=}")
    print(f"{len(tfidf_transcripts_df)=}")
    print(f"{len(tfidf_rest_of_explanations_df)=}")
    print()

    tfidf_df = pd.concat([
        tfidf_title_df,
        tfidf_title_text_df,
        tfidf_transcripts_df,
        tfidf_rest_of_explanations_df,
    ])

    # Add tf-idf scores to the database
    db_utils.insert_tfidfs_into_db(db_path, tfidf_df)

# %% CHECK OUT THE RESULTS
# TODO: Check out some results

