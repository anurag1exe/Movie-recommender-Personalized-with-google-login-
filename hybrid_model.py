from cf_model import CFModel
from cb_model import CBModel

class HybridModel:
    def __init__(self, cf_weight=0.7, cb_weight=0.3):
        self.cf_model = CFModel(n_factors=20, n_epochs=10, lr=0.01, reg=0.02)
        self.cb_model = CBModel()
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.user_history_cache = {}

    def train(self, movies, ratings):
        """
        Train both CF and CB models.
        """
        print("Training CF model...")
        self.cf_model.train(ratings)
        print("Training CB model...")
        self.cb_model.train(movies)
        
        # Cache user histories for fast CB prediction
        print("Caching user histories...")
        self.user_history_cache = {}
        for r in ratings:
            uid = r['userId']
            if uid not in self.user_history_cache:
                self.user_history_cache[uid] = {'movies': [], 'ratings': []}
            self.user_history_cache[uid]['movies'].append(r['movieId'])
            self.user_history_cache[uid]['ratings'].append(r['rating'])

    def predict(self, user_id, movie_id):
        """
        Predict rating using weighted combination of CF and CB.
        """
        cf_pred = self.cf_model.predict(user_id, movie_id)
        
        if user_id in self.user_history_cache:
            history = self.user_history_cache[user_id]
            cb_pred = self.cb_model.predict(history['movies'], history['ratings'], movie_id)
        else:
            cb_pred = 3.0 # Fallback for new user
            
        hybrid_pred = (cf_pred * self.cf_weight) + (cb_pred * self.cb_weight)
        return max(0.5, min(5.0, hybrid_pred))

    def recommend(self, user_id, all_movie_ids, top_n=10):
        """
        Rank all movies for a given user.
        """
        predictions = []
        
        # Determine movies already seen by user to exclude them
        seen_movies = set()
        if user_id in self.user_history_cache:
            seen_movies = set(self.user_history_cache[user_id]['movies'])
            
        for m_id in all_movie_ids:
            if m_id in seen_movies:
                continue
            pred = self.predict(user_id, m_id)
            predictions.append((m_id, pred))
            
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

    def save(self, cf_path, cb_path):
        self.cf_model.save(cf_path)
        self.cb_model.save(cb_path)

    def load(self, cf_path, cb_path, ratings=None):
        self.cf_model.load(cf_path)
        self.cb_model.load(cb_path)
        
        if ratings is not None:
            for r in ratings:
                uid = r['userId']
                if uid not in self.user_history_cache:
                    self.user_history_cache[uid] = {'movies': [], 'ratings': []}
                self.user_history_cache[uid]['movies'].append(r['movieId'])
                self.user_history_cache[uid]['ratings'].append(r['rating'])
