import ast
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('data/keys.env')

# Credentials
USE_FP16 = str(os.getenv("USE_FP16")).lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL")

engine = None  # Initialize the database engine variable

@lru_cache(maxsize = 4)
def get_model(model):
    return SentenceTransformer(model)

def initialize_database():
    """
    Initializes the database connection.
    """
    global engine
    if engine is None:
        # Connect to a supabase database (or any other database of your choice)
        engine = create_engine(
            DATABASE_URL,
            connect_args={"prepare_threshold": None},
            pool_pre_ping=True,
        )

def extract_from_database(table_name):
    """
    Extracts data from the database.

    Args:
        table_name (str): The name of the table to extract data from.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted data.
    """
    initialize_database()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)
            return df
    except Exception as e:
        print(f"Error extracting data from database => Exception: {e}")
        return None

def load(df, file_path, format = "csv"):
    """
    Loads a DataFrame to a specified file format (CSV or JSON).

    Args:
        df (pd.DataFrame): The DataFrame to load.
        file_path (str): The path to the output file.
        format (str): The format to load the DataFrame into ("csv" or "json").

    Raises:
        ValueError: If the specified format is not supported.
    """
    if format == "json":
        try:
            df.to_json(file_path, orient = "records", lines = False)
            assert os.path.exists("data/etl/books_database_csv_etl/results/books_database.json"), "JSON file was not created."
        except Exception as e:
            print(f"Error loading data to JSON => Exception: {e}")
    elif format == "csv":
        try:
            df.to_csv(file_path, index = False)
            assert os.path.exists("data/etl/books_database_csv_etl/results/books_database.csv"), "CSV file was not created."
        except Exception as e:
            print(f"Error loading data to CSV => Exception: {e}")
    else:
        raise ValueError("Unsupported format. Please use 'csv' or 'json'.")

def list_from_relation(df, relation_df, other_id, other_column):
    """
    Creates a dictionary mapping book IDs to lists of related values from another DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing book IDs.
        relation_df (pd.DataFrame): The DataFrame containing related values.
        other_id (str): The column name in relation_df that corresponds to the related ID.
        other_column (str): The column name in relation_df that contains the related values.

    Returns:
        dict: A dictionary mapping book IDs to lists of related values.
    """
    result = {}
    for _, row in df.iterrows():
        book_id = row["book_id"]
        related_id = row[other_id]
        related_values = relation_df[relation_df[other_id] == related_id][other_column].values[0]
        if book_id not in result:
            result[book_id] = []
        result[book_id].append(related_values)
    return result

def parse_embedding(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            result = ast.literal_eval(value.strip())
            return result if isinstance(result, list) else []
        except (SyntaxError, ValueError):
            return []
    return []

def transform_to_embeddings(chunks):
    """
    Transforms a list of text chunks into embeddings using a pre-trained model.

    Args:
        chunks (list): A list of text chunks.

    Returns:
        numpy.ndarray: A 2D array where each row corresponds to the embedding of a chunk.
    """
    model_lm12 = get_model("sentence-transformers/all-MiniLM-L6-v2")
    model_mpnet = get_model("sentence-transformers/all-mpnet-base-v2")
    all_embeddings = []
    try:
        embeddings_lm12 = model_lm12.encode(chunks, batch_size = 32, show_progress_bar = True)
        all_embeddings.append(embeddings_lm12.tolist())

        embeddings_mpnet = model_mpnet.encode(chunks, batch_size = 32, show_progress_bar = True)
        all_embeddings.append(embeddings_mpnet.tolist())

        return all_embeddings
    except Exception as e:
        print(f"Error transforming chunks to embeddings: {e}")
        return None

def extract_text_chunks(words, chunk_size = 500, overlap = 50):
    try:
        chunks = []
        step = max(chunk_size - overlap, 1)  # Ensure step is at least 1 to avoid infinite loop
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i:i + chunk_size])
            if(len(chunk_text) > 0):
                chunks.append(chunk_text)
        return chunks
    except Exception as e:
        print(f"Error extracting chunks from {words}: {e}")
        return []

desired_columns = [
    "book_id",
    "title",
    "chapters",
    "topics",
    "related_subjects",
    "authors",
    "description",
    "average_rating",
    "num_ratings",
    "is_available",
    "embedding_representation_minilm",
    "embedding_representation_mpnet",
]

books = extract_from_database("public.books")
authors = extract_from_database("public.authors")
topics = extract_from_database("public.topics")
related_subjects = extract_from_database("public.related_subjects")
description_embeddings = extract_from_database("vector_schema.book_description_embeddings")
books_authors = extract_from_database("public.books_authors")
books_topics = extract_from_database("public.books_topics")
books_related_subjects = extract_from_database("public.books_related_subjects")
book_chapters = extract_from_database("public.book_chapter_titles")

# for each book_id in books_authors, get the corresponding author name from authors table and column "author" and save it as a list
book_authors = list_from_relation(books_authors, authors, "author_id", "author")

# same for books_topics
book_topics = list_from_relation(books_topics, topics, "topic_id", "topic")

# same for books_related_subjects
book_related_subjects = list_from_relation(books_related_subjects, related_subjects, "related_subject_id", "related_subject")

books["authors"] = books["book_id"].apply(lambda x: book_authors.get(x, []))
books["topics"] = books["book_id"].apply(lambda x: book_topics.get(x, []))
books["related_subjects"] = books["book_id"].apply(lambda x: book_related_subjects.get(x, []))
chapters = books["book_id"].apply(lambda x: book_chapters[book_chapters["book_id"] == x]["chapter_title"].tolist())
chapter_level = books["book_id"].apply(lambda x: book_chapters[book_chapters["book_id"] == x]["chapter_level"].tolist())
books["chapters"] = [
    [f"{title} (level {level})" for title, level in zip(titles, levels)]
    for titles, levels in zip(chapters, chapter_level)
]

# result = pd.merge(books, description_embeddings, on = "book_id", how = "outer")
result = books
result['is_available'] = True  # Add a new column "is_available" and set it to True for all rows
result = result.rename(
    columns = {
        "embedding_minilm": "embedding_representation_minilm",
        "embedding_mpnet": "embedding_representation_mpnet",
        "book_name": "title",
        "book_average_rating": "average_rating",  
        }
    )

result["embedding_representation_minilm"] = None
result["embedding_representation_mpnet"] = None

for idx, row in result.iterrows():
    to_be_embedded = (
        f"Input text:\n"
        f"Title: {row['title']}\n"
        f"Description: {row['description']}\n"
        f"Chapters: {', '.join(row['chapters'])}\n"
        f"Topics: {', '.join(row['topics'])}\n"
        f"Related Subjects: {', '.join(row['related_subjects'])}\n"
        f"Authors: {', '.join(row['authors'])}"
    )

    chunks = extract_text_chunks(to_be_embedded.split())
    embedded = transform_to_embeddings(chunks)
    if embedded is None:
        continue

    # list of vectors in each cell
    result.at[idx, "embedding_representation_minilm"] = embedded[0]
    result.at[idx, "embedding_representation_mpnet"] = embedded[1]


result = result[desired_columns]  # Reorder the columns to match the desired order and keep only the desired columns

result["embedding_representation_minilm"] = (result["embedding_representation_minilm"].apply(parse_embedding))

result["embedding_representation_mpnet"] = (result["embedding_representation_mpnet"].apply(parse_embedding))

load(result, "data/etl/books_database_csv_etl/results/books_database.json", format = "json")
load(result, "data/etl/books_database_csv_etl/results/books_database.csv", format = "csv")
