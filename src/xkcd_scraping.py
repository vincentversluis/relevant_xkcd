# %% HEADER
# A getter for explain xkcd data, aimed at delivering parsed data to the user

# %% IMPORTS
import re

import arrow
from bs4 import BeautifulSoup
from dateutil import parser
import requests


# %% FUNCTIONS
def get_explain_xkcd(xkcd_id: int) -> dict:
    """Get the explain xkcd data for a given xkcd id.

    Args:
        xkcd_id (int): The xkcd id.

    Returns:
        dict: Parsed data for the xkcd explanation.
    """
    # Fetch the page and prepare for parsing
    url = f"https://www.explainxkcd.com/wiki/index.php/{xkcd_id}"
    response = requests.get(url)
    response.raise_for_status()  # raise an error if request failed
    soup = BeautifulSoup(response.text, "html.parser")

    # Get title
    title = re.sub(rf"{xkcd_id}: ", "", soup.find("h1").get_text(strip=True))

    # Get date from date button and parse to YYYY-MM-DD
    date_button = soup.find(attrs={"title": "Open this comic on xkcd.com"})
    date_button_text = date_button.get_text().replace("\xa0", " ")
    pattern = rf"#{xkcd_id} \((?P<date>.*)\)"
    date_button_text_re = re.search(pattern, date_button_text)
    date = date_button_text_re.group("date")
    date = parser.parse(date)
    date = arrow.get(date).format("YYYY-MM-DD")

    # Find title text, navigate to actual text and extract
    title_text_ele = soup.find(attrs={"title": "Title text"})
    title_text = re.sub(
        r"Title\xa0text: ", "", title_text_ele.parent.parent.parent.text
    )

    contents = {
        "xkcd_id": xkcd_id,
        "title": title,
        "date": date,
        "title_text": title_text,
        "body": {},  # Fill in by looping through tags
    }

    # Find explanation text
    explanation_text = soup.find_all("div", {"class": "mw-parser-output"})[0]

    # Loop through text to find headings and add contents from tag text
    current_heading = ""
    current_tag_id = 0
    for tag in explanation_text:
        # Find current heading, update on new heading
        if re.match(r"^<h2>.*</h2>$", str(tag)):
            current_heading = re.sub(r"\[.*\]", "", tag.get_text(strip=True))
            contents["body"][current_heading] = {}
            current_tag_id = 0  # Reset tag id

        # Add regular text to current heading
        if current_heading and re.match(r"^(<p>|<dl>).*", str(tag)):
            text = [text for text in re.split(r"[^\S ]", tag.text) if text]
            for fragment in text:
                contents["body"][current_heading][current_tag_id] = fragment
                current_tag_id += 1

        # Add table contents to heading
        if re.match(r'^<table class="wikitable">.*', str(tag)):
            # Find all table cell elements and extract text
            cells = tag.find_all("td")
            for cell in cells:
                text = [cell for cell in re.split(r"[^\S ]", cell.text) if cell]
                for fragment in text:
                    contents["body"][current_heading][current_tag_id] = fragment
                    current_tag_id += 1

    return contents


# %%
