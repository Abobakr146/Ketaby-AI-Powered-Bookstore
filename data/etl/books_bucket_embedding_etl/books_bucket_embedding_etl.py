import pandas as pd
import os
import json
from sqlalchemy import create_engine, text
from b2sdk.v2 import *
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import pymupdf  # PyMuPDF
from dotenv import load_dotenv

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
model_bge_m3 = None  # Initialize the model variable

def initialize_models():
    global model_lm12, model_mpnet, model_bge_m3
    if model_lm12 is None:
        model_lm12 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Load the pre-trained model
    if model_mpnet is None:
        model_mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Load the pre-trained model
    if model_bge_m3 is None:
        model_bge_m3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=USE_FP16)  # Load the pre-trained model, use_fp16=True to use half precision for faster inference and lower memory usage

def initialize_database():
    global engine
    if engine is None:
        engine = create_engine(DATABASE_URL)  # Connect to a supabase database (or any other database of your choice)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM public.books"))
            conn.execute(text("DELETE FROM vector_schema.book_vectors"))

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
    def __init__(self, isbn = 0, name = "", author = "", stored_in = ""):
        self.name = name
        self.isbn = isbn
        self.author = author
        self.stored_in = stored_in

def extract_chunks(file_path, chunk_size=500, overlap=50):
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

def transform_to_embeddings(chunks):
    """
    Transforms a list of text chunks into embeddings using a pre-trained model.

    Args:
        chunks (list): A list of text chunks.

    Returns:
        numpy.ndarray: A 2D array where each row corresponds to the embedding of a chunk.
    """

    all_embeddings = []
    try:
        embeddings_lm12 = model_lm12.encode(chunks, batch_size = 32, show_progress_bar = True)
        all_embeddings.append(embeddings_lm12.tolist())

        embeddings_mpnet = model_mpnet.encode(chunks, batch_size = 32, show_progress_bar = True)
        all_embeddings.append(embeddings_mpnet.tolist())

        output = model_bge_m3.encode(
            chunks,
            return_dense = True,
            return_sparse = True,
            return_colbert_vecs = False,
            batch_size = 32
        )

        dense_vecs = output.get("dense_vecs")

        if dense_vecs is None:
            raise ValueError("BGE-M3 output did not include dense_vecs")
        
        # Get the first (and only) dense vectors
        if isinstance(dense_vecs, list) and len(dense_vecs) == 1:
            dense_vecs = dense_vecs[0]
        
        # Convert to list if it's a numpy array or similar
        if hasattr(dense_vecs, "tolist"):
            dense_vecs = dense_vecs.tolist() 

        # If it's a 1D vector (single chunk), wrap it
        if len(chunks) == 1 and isinstance(dense_vecs, list) and not isinstance(dense_vecs[0], (list, tuple)):
            dense_vecs = [dense_vecs]  # Wrap in a list to maintain consistent shape

        if len(dense_vecs) != len(chunks):
            raise ValueError(f"Number of dense vectors ({len(dense_vecs)}) does not match number of chunks ({len(chunks)})")

        dense_vectors = dense_vecs

        all_embeddings.append(dense_vectors)

        lexical_weights = output.get("lexical_weights")
        formatted_sparse = []
        for lw in lexical_weights:
            cleaned = {str(key):float(value) for key, value in lw.items()}
            formatted_sparse.append(json.dumps(cleaned))

        if len(formatted_sparse) != len(chunks):
            formatted_sparse = [json.dumps({})] * len(chunks)

        all_embeddings.append(formatted_sparse)

        return all_embeddings
    except Exception as e:
        print(f"Error transforming chunks to embeddings: {e}")
        return None

def load_book(book_df):
    """
    Loads a book's information into the database.

    Args:
        book_df (pd.DataFrame): The DataFrame containing the book's information.

    Returns:
        int: The id of the loaded book.
    """

    try:
        with engine.begin() as conn:
            book_df.to_sql('books', con = conn, schema = 'public', if_exists = 'append', index = False)
            print(f"Book '{book_df['book_name'][0]}' loaded into the database.")

            # return the book's id from the database
            name = str(book_df["book_name"].iloc[0])
            result = conn.execute(
                text("SELECT book_id FROM public.books WHERE book_name = :name ORDER BY book_id DESC LIMIT 1"),{"name": name},
            )
            book_id = result.scalar_one_or_none()
            return book_id
    except Exception as e:
        print(f"Error loading book '{book_df['book_name'][0]}' into the database: {e}")
        return None

def load_embeddings(embeddings_df, book_name):
    """
    Loads embeddings into the database.

    Args:
        embeddings_df (pd.DataFrame): The DataFrame containing the embeddings.

    Returns:
        None
    """

    try:
        data = []
        for _, row in embeddings_df.iterrows():
            data.append({
                "book_id": row["book_id"],
                "chunk_title": row["chunk_title"],
                "chunk_text": row["chunk_text"],
                "embedding_minilm": str(row["embedding_minilm"]),
                "embedding_mpnet": str(row["embedding_mpnet"]),
                "embedding_bge_dense": str(row["embedding_bge_dense"]),
                "embedding_bge_sparse": row["embedding_bge_sparse"],
            })
        with engine.begin() as conn:
            conn.execute(
              text("""
                INSERT INTO vector_schema.book_vectors 
                (book_id, chunk_title, chunk_text, embedding_minilm, embedding_mpnet, embedding_bge_dense, embedding_bge_sparse) 
                VALUES (
                    :book_id, 
                    :chunk_title, 
                    :chunk_text, 
                    CAST(:embedding_minilm AS vector), 
                    CAST(:embedding_mpnet AS vector), 
                    CAST(:embedding_bge_dense AS vector), 
                    CAST(:embedding_bge_sparse AS jsonb)
                )
            """), 
                data,
            )
        print(f"Embeddings for book '{book_name}' loaded into the database.")
    except Exception as e:
        print(f"Error loading embeddings for book '{book_name}' into the database: {e}")

def process_books(storage, download_url):
    """
    Extracts books from the given storage bucket and processes them into chunks.

    Args:
        storage (b2sdk.v2.Bucket): The Backblaze B2 storage bucket object.
        download_url (str): The download URL for the storage bucket.

    Returns:
        None
    """

    # load books from backblaze storage bucket
    os.makedirs('raw', exist_ok = True)
    for file_version, folder_name in storage.ls(recursive = True): # recursive = True to get all files in the bucket
        # recursive = False to get only the files in the root directory of the bucket
        if file_version is None or not file_version.file_name.endswith('.pdf'):
            continue
        
        remote_name = file_version.file_name
        local_name = os.path.basename(remote_name)

        # put the file in the raw folder
        local_target_path = os.path.join('raw', local_name)

        print(f"\n[1/5] => Downloading {remote_name} to {local_target_path}")
        download_dest = storage.download_file_by_name(remote_name)
        download_dest.save_to(local_target_path)
        
        
        book = Book() # create a book object to store the book's information

        # get book's link in the bucket
        book.stored_in = f"{download_url}/file/{storage.name}/{file_version.file_name}"
        # get book's name and author from the file name
        book.name = local_name.split('.')[0]

        book_df = pd.DataFrame(
            {
                "isbn": [book.isbn],
                "book_name": [book.name],
                "author": [book.author],
                "stored_in": [book.stored_in],
            }
        )

        print(f"[2/5] => Loading book '{book.name}' into the database")
        book_id = load_book(book_df) # load the book's information into the database

        try:
            print(f"[3/5] => Extracting chunks from {local_target_path}")
            chunks = extract_chunks(local_target_path) # get the chunks from the pdf file

            print(f"[4/5] => Transforming {len(chunks)} chunks into embeddings")
            
            embeddings = transform_to_embeddings(chunks) # transform the chunks into embeddings
            embeddings_df = pd.DataFrame(
                {
                    "book_id": [book_id] * len(chunks),
                    "chunk_title": [f"Chunk {i+1}" for i in range(len(chunks))],
                    "chunk_text": chunks,
                    "embedding_minilm": [embeddings[0][i] for i in range(len(chunks))],
                    "embedding_mpnet": [embeddings[1][i] for i in range(len(chunks))],
                    "embedding_bge_dense": [embeddings[2][i] for i in range(len(chunks))],
                    "embedding_bge_sparse": [embeddings[3][i] for i in range(len(chunks))],  # Store the same sparse weights for each chunk
                }
            )

            print(f"[5/5] => Loading embeddings into supabase pgvector")
            load_embeddings(embeddings_df, book.name) # load the embeddings into the database
        except Exception as e:
            print(f"Error extracting chunks from {local_target_path}: {e}")

def main():
    initialize_models()  # Initialize the models
    initialize_database()  # Initialize the database connection
    bucket_object = get_b2_bucket()  # Get the B2 bucket object
    bucket = bucket_object[0]  # Get the bucket object from the returned list
    download_url = bucket_object[1]  # Get the download URL from the returned list
    process_books(bucket, download_url)  # Process the books in the bucket

if __name__ == "__main__":
    main()