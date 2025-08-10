# %% HEADER
# Some functions to interact with the database
# TODO: Return getters as a list of named tuples
# TODO: Docstrings

# %% IMPORTS
import sqlite3
from collections import namedtuple
import pandas as pd

# %% CLASSES
XkcdProperty = namedtuple("XkcdProperty", ["xkcd_id", "title", "date", "title_text"])
XkcdExplanation = namedtuple(
    "XkcdExplanation", ["xkcd_id", "heading", "tag_id", "text"]
)


# %% FUNCTIONS
def clean_inputs(text: str) -> str:
    """Clean strings for insertion into the database. Swap double quotes for single quotes and such.

    Args:
        text (str): The string to clean

    Returns:
        str: The cleaned string
    """
    text = text.replace('"', "'")
    return text


def create_tables(db_path: str) -> None:
    """Initialise database tables:
    - XKCD_PROPERTIES: General properties of each xkcd
    - XKCD_EXPLAINED: Explanations from explainxkcd.com for each xkcd
    - XKCD_EXPLAINED_TFIDF: TF-IDF vectors for each explained xkcd

    Args:
        db_path (str): path to the database
    """
    conn = sqlite3.connect(db_path)

    # Initialise tables
    conn.execute("""
        CREATE TABLE XKCD_PROPERTIES (
            xkcd_id INTEGER,
            title VARCHAR(256),
            date VARCHAR(256),
            title_text VARCHAR(1024),
            modified_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(xkcd_id)
        );
    """)

    conn.execute("""
        CREATE TABLE XKCD_EXPLAINED (
            xkcd_id INTEGER,
            heading VARCHAR(256),
            tag_id INTEGER,
            text VARCHAR(4096),
            modified_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(xkcd_id, heading, tag_id)
        );
    """)

    conn.execute("""
        CREATE TABLE XKCD_EXPLAINED_TFIDF (
            xkcd_id INTEGER,
            heading VARCHAR(256),
            token VARCHAR(512),
            tfidf FLOAT,
            modified_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(xkcd_id, heading, token)
        );
    """)


def insert_xkcd_properties_into_db(db_path: str, xkcd_properties: dict) -> None:
    """Insert the properties of an xkcd into the database.

    Args:
        db_path (str): The path to the database
        properties (dict): A dict containing the properties of the xkcd in the `xkcd_id`,
            `title`, `date` and `title_text` keys
    """
    conn = sqlite3.connect(db_path)
    conn.execute(f"""
        INSERT OR IGNORE INTO XKCD_PROPERTIES 
        (
                xkcd_id
            ,   title
            ,   date
            ,   title_text
        ) 
        VALUES 
        (
                {xkcd_properties["xkcd_id"]}
            ,   "{clean_inputs(xkcd_properties["title"])}"
            ,   "{xkcd_properties["date"]}"
            ,   "{clean_inputs(xkcd_properties["title_text"])}"
        )
        """)
    # Commit and close
    conn.commit()
    conn.close()


def insert_xkcd_explained_into_db(db_path: str, xkcd_explained: dict) -> None:
    """Insert the explanation of an xkcd into the database.

    Args:
        db_path (str): The path to the database
        xkcd_explained (dict): A dict containing the properties of the xkcd in the
            `body` key, with each heading as a key and each tag as a list of texts
    """
    xkcd_id = xkcd_explained["xkcd_id"]

    conn = sqlite3.connect(db_path)
    # Insert each text one by one
    for heading, tags in xkcd_explained["body"].items():
        for tag_id, text in tags.items():
            conn.execute(f"""
                INSERT OR IGNORE INTO XKCD_EXPLAINED 
                (
                        xkcd_id
                    ,   heading
                    ,   tag_id
                    ,   text
                ) 
                VALUES 
                (
                        {xkcd_id}
                    ,   "{clean_inputs(heading)}"
                    ,   "{tag_id}"
                    ,   "{clean_inputs(text)}"
                )
                """)
    # Commit and close
    conn.commit()
    conn.close()


def insert_tfidfs_into_db(db_path: str, tfidf_df: pd.DataFrame) -> None:
    """Insert calculated tf-idf scores into the database.

    Args:
        db_path (str): The path to the database
        tfidfs (dict): A dict containing the properties of the xkcd in the
            `body` key, with each heading as a key and each tag as a list of texts
    """
    conn = sqlite3.connect(db_path)
    tfidf_df.to_sql("XKCD_EXPLAINED_TFIDF", conn, if_exists="append", index=False)


def get_xkcd_properties(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    result = pd.read_sql("SELECT * FROM XKCD_PROPERTIES", conn)
    return result


def get_xkcd_explained(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    result = pd.read_sql("SELECT * FROM XKCD_EXPLAINED", conn)
    return result


def get_scraped_xkcd_ids(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    result = pd.read_sql("SELECT DISTINCT xkcd_id FROM XKCD_PROPERTIES", conn)[
        "xkcd_id"
    ].to_list()
    return result


# %% MAIN
# Initialise database
# create_tables('../../data/relevant_xkcd.db')

# %%
