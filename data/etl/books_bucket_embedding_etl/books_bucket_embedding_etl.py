import pandas as pd
import os
import json
from sqlalchemy import create_engine, text, bindparam
from b2sdk.v2 import *
from sentence_transformers import SentenceTransformer
# from FlagEmbedding import BGEM3FlagModel
import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from collections import defaultdict
from functools import lru_cache

# Load environment variables
load_dotenv()

# Credentials
KEY_ID = os.getenv("B2_KEY_ID")
APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
BUCKET_NAME = os.getenv("B2_BUCKET_NAME")
USE_FP16 = str(os.getenv("USE_FP16")).lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL")

engine = None  # Initialize the database engine variable
model_lm12 = None  # Initialize the model variable
model_mpnet = None  # Initialize the model variable
# model_bge_m3 = None  # Initialize the model variable

def extract_titles(pdf_path: str):
    try:
        with pymupdf.open(pdf_path) as doc:
            toc = doc.get_toc() # list of [level, title, page]
            if toc:
                return {
                    "source": "bookmarks",
                    "items": [
                        {
                            "level": level,
                            "title": title.strip(),
                            "page": page
                        }
                        for level, title, page in toc
                    ]
                }
            return {
                "source": "none",
                "items": [],
            }
    except Exception as e:
        print(f"Failed to extract titles from {pdf_path}: {e}")
        return {
                "source": "error",
                "items": [],
        }

def extract_load_titles(book, book_id):
    try:
        toc = extract_titles(book)
        if toc['items']:
            items = toc.get('items', [])
            rows = [
                {
                    'book_id': book_id,
                    **item
                }
                for item in items
            ]
            insert_stmt = text("""
                INSERT INTO public.book_chapter_titles (book_id, chapter_title, chapter_level, chapter_page_number)
                VALUES (:book_id, :title, :level, :page)
                ON CONFLICT (chapter_title_id) DO NOTHING
            """)
            with engine.begin() as conn:
                conn.execute(insert_stmt, rows)
    except Exception as e:
        print(f"Error loading book chapter titles '{book}' into the database: {e}")

def initialize_models():
    global model_lm12, model_mpnet, model_bge_m3
    if model_lm12 is None:
        model_lm12 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Load the pre-trained model
    if model_mpnet is None:
        model_mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Load the pre-trained model
#    if model_bge_m3 is None:
#        model_bge_m3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=USE_FP16)  # Load the pre-trained model, use_fp16=True to use half precision for faster inference and lower memory usage

@lru_cache(maxsize = 4)
def get_model(model):
    return SentenceTransformer(model)

def initialize_database():
    global engine
    if engine is None:
        # Connect to a supabase database (or any other database of your choice)
        engine = create_engine(
            DATABASE_URL,
            connect_args={"prepare_threshold": None},
            pool_pre_ping=True,
        )
        # with engine.begin() as conn:
            # conn.execute(text("DELETE FROM public.books"))
            # conn.execute(text("DELETE FROM vector_schema.book_vectors"))

def get_b2_bucket():
    # Authorize and connect to B2
    bucket_and_download_url = []
    info = InMemoryAccountInfo()
    b2_api = B2Api(info) # Create a B2Api instance with the account info
    b2_api.authorize_account("production", KEY_ID, APPLICATION_KEY)

    bucket_and_download_url.append(b2_api.get_bucket_by_name(BUCKET_NAME))
    bucket_and_download_url.append(b2_api.account_info.get_download_url())

    return bucket_and_download_url

class Book:
    def __init__(self, isbn_electronic = "0", name = "", stored_in = "", cover_url = "", description = ""):
        self.name = name
        self.isbn_electronic = isbn_electronic
        self.stored_in = stored_in
        self.cover_url = cover_url
        self.description = description

def extract_chunks(file_path, chunk_size = 500, overlap = 50):
    """
    Extracts text from a PDF file and splits it into chunks of specified size with overlap.
    
    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of words to overlap between chunks.

    Returns:
        list: A list of text chunks.
    """

    try:
        doc = pymupdf.open(file_path) # Open the PDF file
        full_text = " ".join([page.get_text() for page in doc]) # Extract text from all pages
        # the " " is important to avoid concatenating words from different pages without spaces
        doc.close() # Close the PDF file
        words = full_text.split()
        
        chunks = []
        step = max(chunk_size - overlap, 1)  # Ensure step is at least 1 to avoid infinite loop
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i:i + chunk_size])
            if(len(chunk_text) > 0):
                chunks.append(chunk_text)
        return chunks
    except Exception as e:
        print(f"Error extracting chunks from {file_path}: {e}")
        return []

def extract_description_chunks(words, chunk_size = 500, overlap = 50):
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

def load_list(load, table_name, column_name):
    if not load:
        return pd.DataFrame(columns=[f"{column_name}_id", column_name])
        
    stmt = text(f"""
        INSERT INTO {table_name} ({column_name})
        VALUES (:value)
        ON CONFLICT ({column_name}) DO NOTHING
    """)
    select_stmt = text(f"""
        SELECT *
        FROM {table_name}
        WHERE {column_name} IN :values
    """).bindparams(bindparam("values", expanding=True))
    
    try:
        with engine.begin() as conn:
            conn.execute(
                stmt,
                [{"value": item} for item in load],
            )
            
            return pd.read_sql(select_stmt, conn, params={"values": load})
    except Exception as e:
        print(f"Error loading list to table '{table_name}' => Exception: {e}")
        return pd.DataFrame(columns=[f"{column_name}_id", column_name])

def load_relations(book_id, other_df, table_name):
    if book_id is None or other_df is None or other_df.empty:
        return
        
    first_col = other_df.columns[0]
    df = other_df[[first_col]].copy()
    df["book_id"] = book_id

    stmt = text(f"""
        INSERT INTO {table_name} (book_id, {first_col})
        VALUES (:book_id, :other_id)
        ON CONFLICT (book_id, {first_col}) DO NOTHING
    """)

    data = [{'book_id': row['book_id'], 'other_id': row[first_col]} for _, row in df.iterrows()]

    try:
        with engine.begin() as conn:
            conn.execute(
                stmt,
                data,
            )
    except Exception as e:
        print(f"Error loading relations to table '{table_name}' => Exception: {e}")

def load_book(book_df):
    """
    Loads a book's information into the database if it doesn't already exist,
    and returns its book_id.
    """
    if book_df is None or book_df.empty:
        return None

    row = book_df.iloc[0].to_dict()

    # Adjust the conflict target (e.g., isbn_electronic or book_name) to match a UNIQUE constraint in public.books
    insert_stmt = text("""
        INSERT INTO public.books (isbn_electronic, book_name, stored_in, cover_url, description)
        VALUES (:isbn_electronic, :book_name, :stored_in, :cover_url, :description)
        ON CONFLICT (isbn_electronic) DO NOTHING
    """)

    select_stmt = text("""
        SELECT book_id
        FROM public.books
        WHERE book_name = :book_name AND isbn_electronic = :isbn_electronic
        ORDER BY book_id DESC
        LIMIT 1
    """)

    try:
        with engine.begin() as conn:
            conn.execute(insert_stmt, row)
            book_id = conn.execute(select_stmt, {"book_name": row["book_name"], "isbn_electronic": row['isbn_electronic']}).scalar_one_or_none()
            print(f"Book '{row['book_name']}' ready with book_id={book_id}.")
            return book_id
    except Exception as e:
        print(f"Error loading book '{row.get('book_name', '')}' into the database: {e}")
        return None

def load_embeddings(embeddings_df, book_name):
    """
    Loads embeddings into vector_schema.book_vectors, skipping chunks
    that already exist for the given book_id based on chunk_text.
    """
    if embeddings_df is None or embeddings_df.empty:
        return

    try:
        book_id = int(embeddings_df["book_id"].iloc[0])

        with engine.begin() as conn:
            existing_rows = conn.execute(
                text("""
                    SELECT chunk_text
                    FROM vector_schema.book_vectors
                    WHERE book_id = :book_id
                """),
                {"book_id": book_id},
            ).fetchall()
            existing_texts = {row[0] for row in existing_rows}

            data = []
            seen_in_batch = set()

            for _, row in embeddings_df.iterrows():
                chunk_text = row["chunk_text"]
                if chunk_text in existing_texts or chunk_text in seen_in_batch:
                    continue
                seen_in_batch.add(chunk_text)

                data.append({
                    "book_id": row["book_id"],
                    "chunk_title": row["chunk_title"],
                    "chunk_text": chunk_text,
                    "embedding_minilm": str(row["embedding_minilm"]),
                    "embedding_mpnet": str(row["embedding_mpnet"]),
                })

            if not data:
                print(f"All embeddings for book '{book_name}' already exist; skipped.")
                return

            conn.execute(
                text("""
                    INSERT INTO vector_schema.book_vectors
                    (book_id, chunk_title, chunk_text, embedding_minilm, embedding_mpnet)
                    VALUES (
                        :book_id,
                        :chunk_title,
                        :chunk_text,
                        CAST(:embedding_minilm AS vector),
                        CAST(:embedding_mpnet AS vector)
                    )
                """),
                data,
            )
        print(f"Loaded {len(data)} new embeddings for book '{book_name}' into the database.")
    except Exception as e:
        print(f"Error loading embeddings for book '{book_name}' into the database: {e}")

def load_description_embeddings(embeddings_df, book_name):
    """
    Loads embeddings into the database.

    Args:
        embeddings_df (pd.DataFrame): The DataFrame containing the embeddings.

    Returns:
        None
    """

    if embeddings_df is None or embeddings_df.empty:
        return

    try:
        book_id = int(embeddings_df["book_id"].iloc[0])

        with engine.begin() as conn:
            existing_rows = conn.execute(
                text("""
                    SELECT chunk_text
                    FROM vector_schema.book_description_embeddings
                    WHERE book_id = :book_id
                """),
                {"book_id": book_id},
            ).fetchall()
            existing_texts = {row[0] for row in existing_rows}

            data = []
            seen_in_batch = set()

            for _, row in embeddings_df.iterrows():
                chunk_text = row["chunk_text"]
                if chunk_text in existing_texts or chunk_text in seen_in_batch:
                    continue
                seen_in_batch.add(chunk_text)

                data.append({
                    "book_id": row["book_id"],
                    "chunk_title": row["chunk_title"],
                    "chunk_text": chunk_text,
                    "embedding_minilm": str(row["embedding_minilm"]),
                    "embedding_mpnet": str(row["embedding_mpnet"]),
                })

            if not data:
                print(f"Description embeddings for book '{book_name}' already exist; skipped.")
                return

            conn.execute(
                text("""
                    INSERT INTO vector_schema.book_description_embeddings
                    (book_id, chunk_title, chunk_text, embedding_minilm, embedding_mpnet)
                    VALUES (
                        :book_id,
                        :chunk_title,
                        :chunk_text,
                        CAST(:embedding_minilm AS vector),
                        CAST(:embedding_mpnet AS vector)
                    )
                """),
                data,
            )
        print(f"Loaded {len(data)} new description embeddings for book '{book_name}' into the database.")
    except Exception as e:
        print(f"Error loading description embeddings for book '{book_name}' into the database: {e}")

def process_books(storage, download_url):
    """
    Extracts books from the given storage bucket and processes them into chunks.
    """
    os.makedirs("raw_books", exist_ok=True)
    os.makedirs("raw_metadata", exist_ok=True)

    books_by_key = defaultdict(dict)

    # First pass: group .pdf, _metadata.json, and _cover.jpg files by book key
    for file_version, _ in storage.ls(recursive=True):
        if file_version is None:
            continue

        remote_name = file_version.file_name
        stem, ext = os.path.splitext(remote_name)
        ext = ext.lower()

        if ext == ".pdf":
            books_by_key[stem]["pdf"] = remote_name
        elif ext == ".json" and stem.endswith("_metadata"):
            base_key = stem[: -len("_metadata")]
            books_by_key[base_key]["json"] = remote_name
        elif ext in {".jpg", ".jpeg", ".png"} and stem.endswith("_cover"):
            base_key = stem[: -len("_cover")]
            books_by_key[base_key]["cover"] = remote_name

    # Second pass: process each complete book entry once
    for _, files in books_by_key.items():
        if "pdf" not in files:
            continue

        book = Book()
        authors_df = pd.DataFrame(columns=["author_id", "author"])
        topics_df = pd.DataFrame(columns=["topic_id", "topic"])
        related_subjects_df = pd.DataFrame(columns=["related_subject_id", "related_subject"])

        if "cover" in files:
            book.cover_url = f"{download_url}/file/{storage.name}/{files['cover']}"

        if "json" in files:
            meta_remote_name = files["json"]
            meta_local_target_path = os.path.join(
                "raw_metadata", os.path.basename(meta_remote_name)
            )
            download_meta = storage.download_file_by_name(meta_remote_name)
            download_meta.save_to(meta_local_target_path)

            with open(meta_local_target_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            book.name = metadata.get("title", "")
            book.isbn_electronic = metadata.get("isbn_electronic", "0")
            book.description = metadata.get("description", "")

            authors = [item for item in metadata.get("authors", []) if item]
            topics = [item for item in metadata.get("topics", []) if item]
            related_subjects = [item for item in metadata.get("related_subjects", []) if item]

            authors_df = load_list(authors, "authors", "author")
            topics_df = load_list(topics, "topics", "topic")
            related_subjects_df = load_list(related_subjects, "related_subjects", "related_subject")

        pdf_remote_name = files["pdf"]
        local_target_path = os.path.join("raw_books", os.path.basename(pdf_remote_name))

        print(f"\n[1/5] => Downloading {pdf_remote_name} to {local_target_path}")
        download_dest = storage.download_file_by_name(pdf_remote_name)
        download_dest.save_to(local_target_path)

        book.stored_in = f"{download_url}/file/{storage.name}/{pdf_remote_name}"

        book_df = pd.DataFrame(
            {
                "isbn_electronic": [book.isbn_electronic],
                "book_name": [book.name],
                "stored_in": [book.stored_in],
                "cover_url": [book.cover_url],
                "description": [book.description],
            }
        )

        print(f"[2/5] => Loading book '{book.name}' into the database")
        book_id = load_book(book_df)

        load_relations(book_id, authors_df, "books_authors")
        load_relations(book_id, topics_df, "books_topics")
        load_relations(book_id, related_subjects_df, "books_related_subjects")
        extract_load_titles(local_target_path, book_id)

        try:
            print(f"[3/5] => Extracting chunks from {local_target_path}")
            chunks = extract_chunks(local_target_path)
            description_chunks = extract_description_chunks((book.description or "").split())

            print(f"[4/5] => Transforming {len(chunks)} chunks into embeddings")
            embeddings = transform_to_embeddings(chunks)
            if not embeddings or not chunks:
                continue

            embeddings_df = pd.DataFrame(
                {
                    "book_id": [book_id] * len(chunks),
                    "chunk_title": [f"Chunk {i+1}" for i in range(len(chunks))],
                    "chunk_text": chunks,
                    "embedding_minilm": [embeddings[0][i] for i in range(len(chunks))],
                    "embedding_mpnet": [embeddings[1][i] for i in range(len(chunks))]
                }
            )

            if description_chunks:
                description_embeddings = transform_to_embeddings(description_chunks)
                if description_embeddings:
                    n_desc = len(description_chunks)
                    description_embeddings_df = pd.DataFrame({
                            "book_id": [book_id] * n_desc,
                            "chunk_title": [f"Chunk {i+1}" for i in range(n_desc)],
                            "chunk_text": description_chunks,
                            "embedding_minilm": [description_embeddings[0][i] for i in range(n_desc)],
                            "embedding_mpnet": [description_embeddings[1][i] for i in range(n_desc)],
                    })
                    load_description_embeddings(description_embeddings_df, book.name)

            print("[5/5] => Loading embeddings into supabase pgvector")
            load_embeddings(embeddings_df, book.name)
        except Exception as e:
            print(f"Error processing {local_target_path}: {e}")

def main():
    global model_lm12, model_mpnet
    # initialize_models()  # Initialize the models
    initialize_database()  # Initialize the database connection
    bucket_object = get_b2_bucket()  # Get the B2 bucket object
    bucket = bucket_object[0]  # Get the bucket object from the returned list
    download_url = bucket_object[1]  # Get the download URL from the returned list
    process_books(bucket, download_url)  # Process the books in the bucket

if __name__ == "__main__":
    main()