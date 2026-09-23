import sqlite3
import os
import time

DB_PATH = os.path.join("data", "cinemate.db")

def get_connection():
    if not os.path.exists("data"):
        os.makedirs("data")
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            user_id INTEGER UNIQUE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER,
            movie_id INTEGER,
            rating REAL,
            timestamp INTEGER,
            PRIMARY KEY (user_id, movie_id)
        )
    ''')
    conn.commit()
    conn.close()

def get_or_create_user(email, default_user_id=None):
    """
    Given an email, return their internal user_id.
    If they don't exist, assign them the default_user_id (to link with ML dataset)
    or generate a new high ID.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT user_id FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    if row:
        conn.close()
        return row[0]
    
    # Generate a new unique ID, making sure it's higher than existing MovieLens users
    c.execute("SELECT MAX(user_id) FROM users")
    max_id_row = c.fetchone()
    max_id = max_id_row[0] if max_id_row[0] else 1000000
    
    new_user_id = default_user_id if default_user_id else (max_id + 1)
    
    c.execute("INSERT INTO users (email, user_id) VALUES (?, ?)", (email, new_user_id))
    conn.commit()
    conn.close()
    return new_user_id

def add_rating(user_id, movie_id, rating):
    conn = get_connection()
    c = conn.cursor()
    # Upsert rating
    c.execute('''
        INSERT INTO ratings (user_id, movie_id, rating, timestamp) 
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating, timestamp=excluded.timestamp
    ''', (user_id, movie_id, float(rating), int(time.time())))
    conn.commit()
    conn.close()

def get_user_ratings(user_id):
    """Returns a list of dicts similar to the dataset ratings"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT user_id, movie_id, rating, timestamp FROM ratings WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    
    ratings = []
    for r in rows:
        ratings.append({
            'userId': r[0],
            'movieId': r[1],
            'rating': r[2],
            'timestamp': r[3]
        })
    return ratings

def get_all_db_ratings():
    """Returns all ratings from the local db"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT user_id, movie_id, rating, timestamp FROM ratings")
    rows = c.fetchall()
    conn.close()
    
    ratings = []
    for r in rows:
        ratings.append({
            'userId': r[0],
            'movieId': r[1],
            'rating': r[2],
            'timestamp': r[3]
        })
    return ratings

def merge_ratings(base_ratings, db_ratings):
    """Merges dynamic db ratings into the base ML dataset ratings in-memory."""
    # Create a lookup for fast override
    db_lookup = {(r['userId'], r['movieId']): r for r in db_ratings}
    
    merged = []
    # Add base ratings unless overridden
    for br in base_ratings:
        key = (br['userId'], br['movieId'])
        if key in db_lookup:
            merged.append(db_lookup[key])
            del db_lookup[key] # Remove so we don't duplicate
        else:
            merged.append(br)
            
    # Add remaining db ratings
    merged.extend(db_lookup.values())
    return merged

# Initialize on load
init_db()
