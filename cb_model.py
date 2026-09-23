import numpy as np
import pickle
import math
from collections import defaultdict

class CBModel:
    def __init__(self):
        self.movie_idx_map = {}
        self.idx_movie_map = {}
        self.cosine_sim = None
        self.movies = []

    def train(self, movies):
        """
        Train the content-based model using movie genres.
        movies is a list of dicts: {'movieId': iid, 'title': t, 'genres': g}
        """
        self.movies = movies
        N = len(movies)
        
        # Parse genres and build vocabulary
        movie_terms = []
        df = defaultdict(int)
        
        for idx, m in enumerate(movies):
            self.movie_idx_map[m['movieId']] = idx
            self.idx_movie_map[idx] = m['movieId']
            
            genres = m['genres'].split('|') if m['genres'] != '(no genres listed)' else []
            genres = [g.lower().strip() for g in genres]
            
            tf = defaultdict(int)
            for g in genres:
                tf[g] += 1
                
            for g in set(genres):
                df[g] += 1
                
            movie_terms.append(tf)
            
        # Compute TF-IDF vectors
        vocab = list(df.keys())
        vocab_idx = {word: i for i, word in enumerate(vocab)}
        V = len(vocab)
        
        tfidf_matrix = np.zeros((N, V))
        for idx, tf in enumerate(movie_terms):
            for word, count in tf.items():
                if word in vocab_idx:
                    v_idx = vocab_idx[word]
                    # tf-idf formula
                    tf_val = count
                    idf_val = math.log((1 + N) / (1 + df[word])) + 1
                    tfidf_matrix[idx, v_idx] = tf_val * idf_val
                    
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        tfidf_matrix = tfidf_matrix / norms
        
        # Compute cosine similarity matrix
        print("Computing cosine similarity matrix...")
        self.cosine_sim = np.dot(tfidf_matrix, tfidf_matrix.T)
        print("CB Model training complete.")

    def predict(self, user_history_movies, user_history_ratings, target_movie_id):
        """
        Predict rating based on weighted average of similar movies the user has rated.
        """
        if target_movie_id not in self.movie_idx_map:
            return 3.0 # Default if movie unseen
            
        target_idx = self.movie_idx_map[target_movie_id]
        
        sim_scores = []
        ratings = []
        for m_id, r in zip(user_history_movies, user_history_ratings):
            if m_id in self.movie_idx_map:
                m_idx = self.movie_idx_map[m_id]
                sim = self.cosine_sim[target_idx][m_idx]
                sim_scores.append(sim)
                ratings.append(r)
                
        sim_scores = np.array(sim_scores)
        ratings = np.array(ratings)
        
        if len(sim_scores) == 0 or np.sum(sim_scores) == 0:
            return 3.0 # Default average rating
            
        # Weighted average of ratings based on similarity
        predicted_rating = np.dot(sim_scores, ratings) / np.sum(sim_scores)
        return max(0.5, min(5.0, predicted_rating))

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            # We don't save the full dense matrix if it's too big, but for 10k movies it's 100M elements (400MB).
            # To be safe and save memory, we can save sparse or just recompute, but saving is fine for now.
            pickle.dump({
                'cosine_sim': self.cosine_sim,
                'movie_idx_map': self.movie_idx_map,
                'idx_movie_map': self.idx_movie_map,
                'movies': self.movies
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.cosine_sim = data['cosine_sim']
            self.movie_idx_map = data['movie_idx_map']
            self.idx_movie_map = data['idx_movie_map']
            self.movies = data['movies']
