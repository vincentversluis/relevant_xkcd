# %% HEADER

# %% IMPORTS
import inflect

# %% INITIALISE
inflect_engine = inflect.engine()


# %% FUNCTIONS
def get_ngram_weight(
    n_gram: str,
    n_gram_weights: dict | str | None = None,
) -> float:
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


def sub_number_to_words(text, numerical_threshold=20):
    num_words = {
        str(i): inflect_engine.number_to_words(i)
        for i in range(1, numerical_threshold + 1)
    }

    # Pattern matches whole numbers between 0 and 20 as standalone words
    pattern = re.compile(r"\b(?:" + "|".join(num_words.keys()) + r")\b")

    return pattern.sub(lambda m: num_words[m.group()], text)
