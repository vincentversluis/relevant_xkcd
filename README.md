# relevant_xkcd

Some coding for getting an xkcd relevant to a given text.

## Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/vincentversluis/relevant_xkcd.git
cd relevant_xkcd
pip install -r requirements.txt
```

## Usage

Use the scripts `/src/1_scrape_xkcd.py` and `/src/2_calculate_tfidf.py` to scrape and calculate tf-idf scores for xkcds. This will fill a database with the relevant rawish data and precalculated tf-idf scores. The `xkcd_recommending.recommend_xkcd` function uses this database to recommend xkcds that are relevant to a given text.

## Example

Use the `recommend_xkcd` function to get xkcds that are relevant to a given text:

```python
from xkcd_recommending import recommend_xkcd

text = "Choosing a password that is both strong and easy to remember"

recommend_xkcd(text, top_n=5)
```

The result is a dataframe that looks like this:

| score | id_title |link
|-------|----------|----
1.40 | 936 - Password Strength | https://www.explainxkcd.com/wiki/index.php/936
0.89 | 1286 - Encryptic | https://www.explainxkcd.com/wiki/index.php/1286
0.82 | 1121 - Identity | https://www.explainxkcd.com/wiki/index.php/1121
0.76 | 792 - Password Reuse | https://www.explainxkcd.com/wiki/index.php/792
0.76 | 2176 - How Hacking Works | https://www.explainxkcd.com/wiki/index.php/2176

For some more examples, see the `/example/recommond_xkcd.ipynb` notebook.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
