import os
import tarfile
import urllib.request
import re
import math
import pickle
from collections import defaultdict

class NLPSentimentModel:
    def __init__(self):
        self.vocab = {}
        self.class_word_counts = {0: defaultdict(int), 1: defaultdict(int)}
        self.class_totals = {0: 0, 1: 0}
        self.class_docs = {0: 0, 1: 0}
        self.vocab_size = 0
        self.is_trained = False

    def download_and_extract(self, dest_dir="data"):
        url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        tar_path = os.path.join(dest_dir, "aclImdb_v1.tar.gz")
        extracted_path = os.path.join(dest_dir, "aclImdb")
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            
        if not os.path.exists(extracted_path):
            if not os.path.exists(tar_path):
                print(f"Downloading IMDB dataset from {url}...")
                urllib.request.urlretrieve(url, tar_path)
                print("Download complete.")
            
            print("Extracting dataset...")
            with tarfile.open(tar_path, "r:gz") as tar:
                # To save time, we only extract the 'train' folder
                members = [m for m in tar.getmembers() if m.name.startswith("aclImdb/train/")]
                tar.extractall(path=dest_dir, members=members)
            print("Extraction complete.")
            
        return extracted_path

    def _tokenize(self, text):
        # Very simple tokenizer: lowercase, remove non-alphanumeric, split by space
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text) # remove HTML tags
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def train(self, data_dir="data", max_samples_per_class=2000):
        extracted_path = self.download_and_extract(data_dir)
        train_dir = os.path.join(extracted_path, "train")
        
        print("Training NLP Sentiment Model (Naive Bayes)...")
        # Load and train
        for label, sub_dir in [(1, "pos"), (0, "neg")]:
            dir_path = os.path.join(train_dir, sub_dir)
            if not os.path.exists(dir_path):
                continue
                
            files = os.listdir(dir_path)[:max_samples_per_class]
            for f_name in files:
                with open(os.path.join(dir_path, f_name), 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                tokens = self._tokenize(text)
                self.class_docs[label] += 1
                
                for word in tokens:
                    self.class_word_counts[label][word] += 1
                    self.class_totals[label] += 1
                    self.vocab[word] = True
                    
        self.vocab_size = len(self.vocab)
        self.is_trained = True
        print(f"Training complete. Vocab size: {self.vocab_size}")

    def predict(self, text):
        if not self.is_trained:
            return 0.5 # Neutral if not trained
            
        tokens = self._tokenize(text)
        
        # Calculate log probabilities to avoid underflow
        # P(Class | Document) ~ P(Class) * Product(P(Word | Class))
        # log P(Class | Document) ~ log P(Class) + Sum(log P(Word | Class))
        
        total_docs = self.class_docs[0] + self.class_docs[1]
        if total_docs == 0: return 0.5
        
        scores = {}
        for c in [0, 1]:
            # Prior probability
            scores[c] = math.log(self.class_docs[c] / total_docs)
            
            for word in tokens:
                # Laplace smoothing
                word_count = self.class_word_counts[c].get(word, 0)
                prob = (word_count + 1) / (self.class_totals[c] + self.vocab_size)
                scores[c] += math.log(prob)
                
        # Convert log scores to a confidence probability (0.0 to 1.0)
        # Using softmax
        max_score = max(scores[0], scores[1])
        exp_0 = math.exp(scores[0] - max_score)
        exp_1 = math.exp(scores[1] - max_score)
        prob_positive = exp_1 / (exp_0 + exp_1)
        
        return prob_positive

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'class_word_counts': self.class_word_counts,
                'class_totals': self.class_totals,
                'class_docs': self.class_docs,
                'vocab_size': self.vocab_size,
                'is_trained': self.is_trained
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.class_word_counts = data['class_word_counts']
            self.class_totals = data['class_totals']
            self.class_docs = data['class_docs']
            self.vocab_size = data['vocab_size']
            self.is_trained = data['is_trained']
