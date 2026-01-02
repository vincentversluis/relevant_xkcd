# relevant_xkcd

Some coding for getting an xkcd relevant to a given text.

## Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/vincentversluis/relevant_xkcd.git
cd relevant_xkcd
pip install -r requirements.txt
```

As this project stores data in a local database, the repository is not intended to be installed as a package. Instead, clone the repository and use the scripts provided.

## Usage

Use the scripts [`/src/1_scrape_xkcd.py`](/src/1_scrape_xkcd.py) to scrape [explainxkcd.com](https://www.explainxkcd.com/wiki/index.php/Main_Page) and [`/src/2_calculate_tfidf.py`](/src/2_calculate_tfidf.py) to precalculate the necessary [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectors. These were made with the [VS Code Jupyter code cells](https://code.visualstudio.com/docs/python/jupyter-support-py), so use VS Code with the Jupyter extension for best results.

After setting up the database, use the [`recommend_xkcd`](/src/xkcd_recommending.py#70) function from the [`xkcd_recommending`](/src/xkcd_recommending.py) module to query xkcds that are relevant to a given text.

## Example

Use the [`recommend_xkcd`](/src/xkcd_recommending.py#70) function to get xkcds that are relevant to a given text. Check out its docstring for more information.

```python
from xkcd_recommending import recommend_xkcd

text = "Choosing a password that is both strong and easy to remember"

recommend_xkcd(
    text=text,
    top_n=5,  # The number of xkcds to recommend
    n_gram_max_length=3,  # The maximum length of n-grams to use
    n_gram_weights="length",  # The weighting method to use for n-grams
    split_pattern=None,  # Use the default English stopwords and punctuation as split pattern
    heading_weights=None,  # Use the default weights for the different parts of the explanation
    number_to_words=True,  # Interpret numbers (like 20) as words (like twenty)
    number_to_words_threshold=20,  # The highest number to convert to written out numbers
    )

```

The result is a dataframe that looks like this:

| score | id_title | link
|-------|----------|----
1.40 | 936 - Password Strength | https://www.explainxkcd.com/wiki/index.php/936
0.89 | 1286 - Encryptic | https://www.explainxkcd.com/wiki/index.php/1286
0.82 | 1121 - Identity | https://www.explainxkcd.com/wiki/index.php/1121
0.76 | 792 - Password Reuse | https://www.explainxkcd.com/wiki/index.php/792
0.76 | 2176 - How Hacking Works | https://www.explainxkcd.com/wiki/index.php/2176

For some more examples, see the [`/example/recommend_xkcd.ipynb`](/example/recommend_xkcd.ipynb) notebook. (Best viewed in an IDE or in a browser.)

## How this works

In short:

The main function [`recommend_xkcd`](/src/xkcd_recommending.py#70) uses a [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectorisation of xkcd descriptions as found on [explainxkcd.com](https://www.explainxkcd.com/wiki/index.php/Main_Page) and attempts to find the nearest semantic equivalent of a given text.

In long:

1. Descriptions of xkcds are scraped from explainxkcd.com.
2. The descriptions are chopped into various size n-gram tokens and the tf-idf scores of each token are calculated. The results are stored in a database.
3. The user defined text (what they want to find an xkcd for) is also chopped into n-gram tokens and their semantic equivalents are found using [Google's word2vec](https://code.google.com/archive/p/word2vec/) model, along with their numerical similarity.
4. The database is queried for all tokens and their equivalents, to find in which xkcd descriptions they occur.
5. The weighted product of each returned token is calculated and summed per xkcd to find a score for each xkcd. The highest scoring xkcds are assumed to be the best matches.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
