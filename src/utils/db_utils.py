# %% HEADER
# Some functions to interact with the database.

# %% IMPORTS
from collections import namedtuple
import sqlite3

import pandas as pd

# %% CLASSES
XkcdProperty = namedtuple("XkcdProperty", ["xkcd_id", "title", "date", "title_text"])
XkcdExplanation = namedtuple(
    "XkcdExplanation", ["xkcd_id", "heading", "tag_id", "text"]
)


# %% FUNCTIONS
def clean_inputs(text: str) -> str:
    """Clean strings for insertion into the database.

    Swap double quotes for single quotes and such.

    Args:
        text (str): The string to clean

    Returns:
        str: The cleaned string
    """
    text = text.replace('"', "'")
    return text


def create_tables(db_path: str) -> None:
    """Initialise database tables.

    - XKCD_PROPERTIES: General properties of each xkcd
    - XKCD_EXPLAINED: Explanations from explainxkcd.com for each xkcd
    - XKCD_EXPLAINED_TFIDF: TF-IDF vectors for each explained xkcd.

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

    Properties are inserted into the `XKCD_PROPERTIES` table.

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

    Inserts the explanations into the `XKCD_EXPLAINED` table.

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

    Inserts the tf-idf scores into the `XKCD_EXPLAINED_TFIDF` table.

    Args:
        db_path (str): The path to the database
        tfidfs (dict): A dict containing the properties of the xkcd in the
            `body` key, with each heading as a key and each tag as a list of texts
    """
    conn = sqlite3.connect(db_path)
    tfidf_df.to_sql("XKCD_EXPLAINED_TFIDF", conn, if_exists="append", index=False)


def get_xkcd_properties(db_path: str) -> pd.DataFrame:
    """Get the properties of all xkcds in the database.

    Args:
        db_path (str): The path to the database

    Returns:
        pd.DataFrame: Properties of all xkcds in the database
    """
    conn = sqlite3.connect(db_path)
    result = pd.read_sql("SELECT * FROM XKCD_PROPERTIES", conn)
    return result


def get_xkcd_explained(db_path: str) -> pd.DataFrame:
    """Get the explanations of all xkcds in the database.

    Args:
        db_path (str): The path to the database

    Returns:
        pd.DataFrame: The explanations of all xkcds in the database
    """
    conn = sqlite3.connect(db_path)
    result = pd.read_sql("SELECT * FROM XKCD_EXPLAINED", conn)
    return result


def get_scraped_xkcd_ids(db_path: str) -> list:
    """Get the xkcd ids of all xkcds in the database.

    Args:
        db_path (str): The path to the database

    Returns:
        list: The xkcd ids of all xkcds in the database
    """
    conn = sqlite3.connect(db_path)
    result = pd.read_sql("SELECT DISTINCT xkcd_id FROM XKCD_PROPERTIES", conn)[
        "xkcd_id"
    ].to_list()
    return result


def get_xkcd_tfidfs(db_path: str, tokens: list) -> pd.DataFrame:
    """Get tf-idf scores for a list of tokens.

    Args:
        db_path (str): The path to the database
        tokens (list): The tokens to get tf-idf scores for.

    Returns:
        pd.DataFrame: The tf-idf scores of the tokens.
    """
    conn = sqlite3.connect(db_path)

    # Use parameterized query with IN clause
    placeholders = ",".join("?" for _ in tokens)
    query = f"""
        SELECT 
            xkcd_id
        ,   heading
        ,   token
        ,   tfidf
        FROM XKCD_EXPLAINED_TFIDF 
        WHERE token IN ({placeholders})
        """

    # Read directly into a DataFrame
    df = pd.read_sql_query(query, conn, params=tokens)

    # Close connection
    conn.close()

    return df


def get_xkcd_properties_for_ids(db_path: str, xkcd_ids: list) -> pd.DataFrame:
    """Get properties for a list of xkcd ids.

    Args:
        db_path (str): The path to the database
        xkcd_ids (list): The xkcd ids to get properties for.

    Returns:
        pd.DataFrame: The properties of the xkcds.
    """
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in xkcd_ids)
    query = f"""
        SELECT 
            xkcd_id
        ,   title
        ,   date
        ,   title_text
        FROM XKCD_PROPERTIES
        WHERE xkcd_id IN ({placeholders})
        """

    # Read directly into a DataFrame
    df = pd.read_sql_query(query, conn, params=xkcd_ids)

    # Close connection
    conn.close()
    return df


# %% MAIN
# Initialise database
if __name__ == "__main__":
    create_tables("../../data/relevant_xkcd.db")

# %%
