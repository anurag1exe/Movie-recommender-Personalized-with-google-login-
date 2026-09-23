import numpy as np
import math
from collections import defaultdict

def time_based_split(ratings, test_ratio=0.2):
    """
    Split ratings into train and test based on timestamp.
    Older ratings go to train, newer to test.
    ratings is a list of dicts.
    """
    ratings.sort(key=lambda x: x['timestamp'])
    split_idx = int(len(ratings) * (1 - test_ratio))
    train_ratings = ratings[:split_idx]
    test_ratings = ratings[split_idx:]
    return train_ratings, test_ratings

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Calculate Precision@K and Recall@K for the predictions.
    predictions: list of (user_id, item_id, true_r, est_r)
    """
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est_r in predictions:
        user_est_true[uid].append((est_r, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    precision = sum(precisions.values()) / len(precisions)
    recall = sum(recalls.values()) / len(recalls)
    return precision, recall

def dcg_at_k(scores, k):
    scores = np.asarray(scores, dtype=float)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0

def ndcg_at_k(true_scores, est_scores, k):
    # Sort true_scores by est_scores
    order = np.argsort(est_scores)[::-1]
    true_scores = np.asarray(true_scores)[order]
    
    idcg = dcg_at_k(sorted(true_scores, reverse=True), k)
    if not idcg:
        return 0.0
    return dcg_at_k(true_scores, k) / idcg

def calculate_ndcg(predictions, k=10):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est_r in predictions:
        user_est_true[uid].append((est_r, true_r))
        
    ndcg_scores = []
    for uid, user_ratings in user_est_true.items():
        if len(user_ratings) > 1:
            y_score = [x[0] for x in user_ratings]
            y_true = [x[1] for x in user_ratings]
            
            score = ndcg_at_k(y_true, y_score, k)
            ndcg_scores.append(score)
            
    if not ndcg_scores:
        return 0.0
    return np.mean(ndcg_scores)

def evaluate_model(model, test_ratings):
    """
    Evaluate the hybrid model on the test set.
    """
    predictions = []
    y_true = []
    y_pred = []
    
    print(f"Evaluating on {len(test_ratings)} test samples...")
    for row in test_ratings:
        uid = row['userId']
        iid = row['movieId']
        true_r = row['rating']
        
        est_r = model.predict(uid, iid)
        
        predictions.append((uid, iid, true_r, est_r))
        y_true.append(true_r)
        y_pred.append(est_r)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    precision_10, recall_10 = precision_recall_at_k(predictions, k=10)
    ndcg = calculate_ndcg(predictions, k=10)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': precision_10,
        'Recall@10': recall_10,
        'NDCG': ndcg
    }
    
    return metrics
