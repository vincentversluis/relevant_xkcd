# %% HEADER
# TODO: Rework comments to nicer language

# %% IMPORTS
from functools import cache
from itertools import chain

import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer

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
    """Get tf-idf scores for each document in the dataframe.

    Args:
        df (pd.DataFrame): The documents to calculate tf-idf scores for, with one document per row.
        doc_id_col (str): The name of the column containing the document ids.
        text_col (str): The name of the column containing the text to calculate tf-idf scores for.
        split_pattern (str): The regex pattern to split the text by.
        n_gram_length (int, optional): The length of the n-grams to use. Defaults to 1.

    Returns:
        pd.DataFrame: _description_
    """
    df["tokens"] = df[text_col].apply(
        lambda text: " ".join(
            get_preprocessed_tokens(
                text, n_gram_length=n_gram_length, split_pattern=split_pattern
            )
        )
    )
    df = df.groupby(doc_id_col)["tokens"].agg(" ".join).reset_index()

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=False, tokenizer=lambda x: x.split(), preprocessor=lambda x: x
    )
    tfidf_matrix = vectorizer.fit_transform(df["tokens"])

    # Get non-zero entries
    nonzero_indices = tfidf_matrix.nonzero()
    rows, cols = nonzero_indices
    scores = tfidf_matrix.data

    # Map back to tokens and doc_ids
    tokens = [vectorizer.get_feature_names_out()[i] for i in cols]
    doc_ids = df[doc_id_col].iloc[rows].values

    tfidf_df = pd.DataFrame({doc_id_col: doc_ids, "token": tokens, "tfidf": scores})
    return tfidf_df


def get_ngrams(text: str, n: int) -> list:
    """Return a list of n-grams from the text.

    Args:
        text (str): The text to split into n-grams.
        n (int, optional): The length of the n-grams to return.

    Returns:
        list: The n-grams of the text.
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


def get_up_to_n_grams(
    text: str, split_pattern: str, n_gram_max_length: int = 1
) -> list:
    """Get n-grams from length 1 to n_gram_max_length.

    Args:
        text (str): The text to get n-grams from.
        split_pattern (str): The regex pattern to split the text by.
        n_gram_max_length (int, optional): The maximum length of the n-grams to return. Defaults to 1.

    Returns:
        list: The n-grams of the text.
    """
    tokens = []
    for n_gram_length in range(1, n_gram_max_length + 1):
        tokens.extend(
            get_preprocessed_tokens(
                text, n_gram_length=n_gram_length, split_pattern=split_pattern
            )
        )
    return tokens


@cache
def get_wordnet_pos(tag: str) -> str:
    """Translate a wordnet tag to a part of speech.

    Args:
        tag (str): The wordnet tag to translate.

    Returns:
        str: The part of speech of the tag.
    """
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


def lemmatise_sentence(sentence: str) -> str:
    """Lemmatise a full sentence.

    This works for English text as the wordnet lemmatiser and pos tagger are English specific.

    Args:
        sentence (str): The sentence to lemmatise.

    Returns:
        str: The lemmatised sentence.
    """
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
    """Get preprocessed tokens from a sentence.

    Args:
        text (str): The text to get tokens from.
        split_pattern (str): The regex pattern to split the text by.
        n_gram_length (int, optional): The length of the n-grams to return. Defaults to 1.

    Returns:
        list: The preprocessed tokens.
    """
    text_lemmatised = lemmatise_sentence(text)
    text_split = split_text_by_regex(text_lemmatised, split_pattern=split_pattern)
    tokens = list(
        chain(*[get_ngrams(fragment, n=n_gram_length) for fragment in text_split])
    )
    return tokens


def split_text_by_regex(text: str, split_pattern: str) -> list:
    """Split text by regex pattern.

    This is mainly a helper function for `get_preprocessed_tokens`.

    Args:
        text (str): The text to split.
        split_pattern (str): The regex pattern to split the text by.

    Returns:
        list: The tokens of the split text.
    """
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
