import os
import zipfile
import io
import requests
import csv

DATA_DIR = 'data'
URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
ZIP_EXTRACT_DIR = os.path.join(DATA_DIR, 'ml-latest-small')

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if os.path.exists(ZIP_EXTRACT_DIR):
        print("Data already exists. Skipping download.")
        return

    print(f"Downloading dataset from {URL}...")
    response = requests.get(URL)
    response.raise_for_status()
    print("Extracting dataset...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(DATA_DIR)
    print("Download and extraction complete.")

def load_data():
    download_and_extract()
    
    movies_path = os.path.join(ZIP_EXTRACT_DIR, 'movies.csv')
    ratings_path = os.path.join(ZIP_EXTRACT_DIR, 'ratings.csv')
    
    movies = []
    with open(movies_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies.append({
                'movieId': int(row['movieId']),
                'title': row['title'],
                'genres': row['genres']
            })
            
    ratings = []
    with open(ratings_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings.append({
                'userId': int(row['userId']),
                'movieId': int(row['movieId']),
                'rating': float(row['rating']),
                'timestamp': int(row['timestamp'])
            })
            
    return movies, ratings

if __name__ == "__main__":
    download_and_extract()
