import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import json
from functools import lru_cache
import ast

# Load environment variables
load_dotenv('data/keys.env')

# Credentials
DATABASE_URL = os.getenv("DATABASE_URL")

engine = None  # Initialize the database engine variable

class USER:
    def __init__(self, user_uuid = None, user_name = None, user_goals = None, user_craeted_at = None, user_updated_at = None):
        self.user_uuid = user_uuid
        self.name = user_name
        self.goals = user_goals
        self.created_at = user_craeted_at
        self.updated_at = user_updated_at

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

def extract(path):
    try:
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error occurred while extracting data from {path}: {e}")
        return None

def load_user(user):
    initialize_database()
    data = {
        "user_goals": user.goals, 
        "user_name": user.name
    }
    stmt = text("""
        INSERT INTO public.users (user_goals, user_name)
        VALUES (:user_goals, :user_name)
        ON CONFLICT (user_uuid) DO UPDATE SET user_goals = EXCLUDED.user_goals
        RETURNING user_uuid
    """)
    try:
        with engine.begin() as conn:
            result = conn.execute(stmt, data)
            return result.scalar_one()  # Return the user_uuid of the newly inserted user
    except Exception as e:
        print(f"Error occurred while loading data into the database: {e}")
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

def load_goal_embeddings(embedding_df):
    initialize_database()
    stmt = text("""
        INSERT INTO vector_schema.user_goal_embedding (user_uuid, goal_embedding_minilm, goal_embedding_mpnet)
        VALUES (
            :user_uuid, 
            CAST(:goal_embedding_minilm AS VECTOR),
            CAST(:goal_embedding_mpnet AS VECTOR)
        )
        ON CONFLICT (user_uuid) DO UPDATE SET
            goal_embedding_minilm = EXCLUDED.goal_embedding_minilm,
            goal_embedding_mpnet = EXCLUDED.goal_embedding_mpnet
    """)
    try:
        with engine.begin() as conn:
            conn.execute(stmt, [{
                "user_uuid": row["user_uuid"],
                "goal_embedding_minilm": row["goal_embedding_minilm"],
                "goal_embedding_mpnet": row["goal_embedding_mpnet"]
            } for _, row in embedding_df.iterrows()])
    except Exception as e:
        print(f"Error occurred while loading data into the database: {e}")

def load_relation(df, table_name):
    if df is None or df.empty:
        print(f"No data to load into {table_name}.")
        return None
    
    initialize_database()
    if table_name == "user_favorite_authors":
        stmt = text("""
            INSERT INTO public.user_favorite_authors (user_uuid, author_id)
            VALUES (:user_uuid, :author_id)
            ON CONFLICT (user_uuid, author_id) DO UPDATE SET author_id = EXCLUDED.author_id
        """)
        try:
            with engine.begin() as conn:
                conn.execute(stmt, [{
                    "user_uuid": row["user_uuid"],
                    "author_id": row["author_id"]
                } for _, row in df.iterrows()])
        except Exception as e:
            print(f"Error occurred while loading data into the database: {e}")
    elif table_name == "user_preferred_topics":
        stmt = text("""
            INSERT INTO public.user_preferred_topics (user_uuid, topic_id)
            VALUES (:user_uuid, :topic_id)
            ON CONFLICT (user_uuid, topic_id) DO UPDATE SET topic_id = EXCLUDED.topic_id
        """)
        try:
            with engine.begin() as conn:
                conn.execute(stmt, [{
                    "user_uuid": row["user_uuid"],
                    "topic_id": row["topic_id"]
                } for _, row in df.iterrows()])
        except Exception as e:
            print(f"Error occurred while loading data into the database: {e}")
    else:
        print(f"Unsupported table name: {table_name}")
        return None

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

raw_user_data = extract('data/etl/synthetic_source_data/synthetic_users_ids-v2.json')
for raw_user in raw_user_data:
    user = USER(user_goals = raw_user.get('goals', ""))
    user_uuid = load_user(user)
    if user_uuid is None:
        print(f"Failed to load user: {user.name}")
        continue
    user.user_uuid = user_uuid  # Update the user object with the returned user_uuid

    chunks = extract_text_chunks(user.goals.split(), chunk_size=500, overlap=50)
    embeddings = transform_to_embeddings(chunks)
    embedding_df = pd.DataFrame({
        "user_uuid": [user.user_uuid] * len(embeddings[0]),
        "goal_embedding_minilm": embeddings[0],
        "goal_embedding_mpnet": embeddings[1]
    })
    load_goal_embeddings(embedding_df)
    
    authors = raw_user.get('favorite_authors', [])
    topics = raw_user.get('preffered_topics', [])

    authors_df = pd.DataFrame({
        "user_uuid": [user.user_uuid] * len(authors),
        "author_id": authors
    })
    topics_df = pd.DataFrame({
        "user_uuid": [user.user_uuid] * len(topics),
        "topic_id": topics
    })
    load_relation(authors_df, "user_favorite_authors")
    load_relation(topics_df, "user_preferred_topics")

    raw_user['user_uuid'] = f"{user.user_uuid}"  # Add the user_uuid to the raw user data
    raw_user['goal_embedding_minilm'] = embeddings[0]
    raw_user['goal_embedding_mpnet'] = embeddings[1]

with open('data/etl/synthetic_source_data/synthetic_users_data.json', 'w', encoding='utf-8') as f:
    json.dump(raw_user_data, f, ensure_ascii=False, indent=4)