# %% NOTES
# Preprocessing and tf-idf stuff
# https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/
# https://towardsdatascience.com/pipeline-for-text-data-pre-processing-a9887b4e2db3/
# https://medium.com/@devangchavan0204/complete-guide-to-text-preprocessing-in-nlp-b4092c104d3e

# Pretrained word embeddings
# https://medium.com/@gauravtailor43/googles-trained-word2vec-model-in-python-ab9b2df09af2

# %% HEADER
# Just prutsing

# %% IMPORTS
from xkcd_recommending import recommend_xkcd

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
