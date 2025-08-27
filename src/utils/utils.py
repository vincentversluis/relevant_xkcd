# %% HEADER
# General utility functions.

# %% IMPORTS
import inflect
import re

# %% INITIALISE
inflect_engine = inflect.engine()


# %% FUNCTIONS
def get_ngram_weight(
    n_gram: str,
    n_gram_weights: dict | str | None = None,
) -> float:
    """Get the weight of an n-gram for a given weighting method.

    Args:
        n_gram (str): The n-gram to get the weight of.
        n_gram_weights (dict | str | None, optional): The weighting method to use. Defaults to None.

    Raises:
        ValueError: No key present for n-gram length
        ValueError: Invalid weighting method for n-gram

    Returns:
        float: _description_
    """    
    if n_gram_weights is None:
        return 1
    elif n_gram_weights == "length":
        return len(n_gram.split("_"))
    elif isinstance(n_gram_weights, dict):
        try:
            return n_gram_weights[len(n_gram.split("_"))]
        except KeyError as e:
            raise ValueError(
                f"Did not pass a weight for ngram length {len(n_gram.split('_'))}"
            ) from e
    else:
        raise ValueError(f"Did not pass a valid ngram weighting method for {n_gram}")


def sub_number_to_words(text: str, numerical_threshold: int = 20) -> str:
    """Substitute numerical numbers in string format with written out numbers.

    Args:
        text (str): The number
        numerical_threshold (int, optional): The highest number to convert to written out numbers. Defaults to 20.

    Returns:
        str: The written out number
    """    
    num_words = {
        str(i): inflect_engine.number_to_words(i)
        for i in range(1, numerical_threshold + 1)
    }

    # Pattern matches whole numbers between 0 and 20 as standalone words
    pattern = re.compile(r"\b(?:" + "|".join(num_words.keys()) + r")\b")

    return pattern.sub(lambda m: num_words[m.group()], text)
