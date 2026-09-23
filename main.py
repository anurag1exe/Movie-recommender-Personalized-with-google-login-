import os
from data_loader import load_data
from evaluation import time_based_split, evaluate_model
from hybrid_model import HybridModel

def main():
    print("Loading data...")
    movies, ratings = load_data()
    
    print("Splitting data based on time...")
    train_ratings, test_ratings = time_based_split(ratings, test_ratio=0.2)
    
    print(f"Train size: {len(train_ratings)}, Test size: {len(test_ratings)}")
    
    print("Initializing Hybrid Model...")
    model = HybridModel(cf_weight=0.7, cb_weight=0.3)
    
    print("Training Hybrid Model...")
    model.train(movies, train_ratings)
    
    print("Evaluating Model...")
    metrics = evaluate_model(model, test_ratings)
    
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nSaving model...")
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/cf_model.pkl', 'models/cb_model.pkl')
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
