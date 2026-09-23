import numpy as np
import pickle

class CFModel:
    def __init__(self, n_factors=20, n_epochs=10, lr=0.01, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.global_mean = 3.0
        self.user_biases = {}
        self.item_biases = {}
        self.user_factors = {}
        self.item_factors = {}

    def train(self, ratings):
        """
        Train CF model using Stochastic Gradient Descent.
        ratings is a list of dicts: {'userId': uid, 'movieId': iid, 'rating': r}
        """
        self.global_mean = np.mean([r['rating'] for r in ratings])
        
        # Initialize biases and factors
        for r in ratings:
            uid = r['userId']
            iid = r['movieId']
            if uid not in self.user_biases:
                self.user_biases[uid] = 0.0
                self.user_factors[uid] = np.random.normal(0, 0.1, self.n_factors)
            if iid not in self.item_biases:
                self.item_biases[iid] = 0.0
                self.item_factors[iid] = np.random.normal(0, 0.1, self.n_factors)
                
        # SGD
        for epoch in range(self.n_epochs):
            for r in ratings:
                uid = r['userId']
                iid = r['movieId']
                true_r = r['rating']
                
                pu = self.user_factors[uid]
                qi = self.item_factors[iid]
                bu = self.user_biases[uid]
                bi = self.item_biases[iid]
                
                est_r = self.global_mean + bu + bi + np.dot(pu, qi)
                err = true_r - est_r
                
                # Update biases
                self.user_biases[uid] += self.lr * (err - self.reg * bu)
                self.item_biases[iid] += self.lr * (err - self.reg * bi)
                
                # Update factors
                self.user_factors[uid] += self.lr * (err * qi - self.reg * pu)
                self.item_factors[iid] += self.lr * (err * pu - self.reg * qi)
                
            print(f"CF Epoch {epoch+1}/{self.n_epochs} complete")

    def predict(self, user_id, movie_id):
        est = self.global_mean
        if user_id in self.user_biases:
            est += self.user_biases[user_id]
        if movie_id in self.item_biases:
            est += self.item_biases[movie_id]
        if user_id in self.user_factors and movie_id in self.item_factors:
            est += np.dot(self.user_factors[user_id], self.item_factors[movie_id])
        
        # Clip to 0.5 - 5.0 rating scale
        return max(0.5, min(5.0, est))

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'global_mean': self.global_mean,
                'user_biases': self.user_biases,
                'item_biases': self.item_biases,
                'user_factors': self.user_factors,
                'item_factors': self.item_factors
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.global_mean = data['global_mean']
            self.user_biases = data['user_biases']
            self.item_biases = data['item_biases']
            self.user_factors = data['user_factors']
            self.item_factors = data['item_factors']
