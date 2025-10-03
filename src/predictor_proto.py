import torch
import torch.nn.functional as F
import json
import joblib
import os
import numpy as np
from src.model_arch import DNA_CNN_Upgraded
import torch, torch.nn.functional as F, json, joblib, os, requests
import numpy as np
from src.model_arch import DNA_CNN_Upgraded
from tqdm import tqdm

def download_file_if_not_exists(url, filepath):
    """Downloads a file if it doesn't already exist."""
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {os.path.basename(filepath)} successfully.")

class DeepSeaHybridClassifier:
    def __init__(self, models_path='models/', data_path='data/processed/'):
        print("INFO: Initializing DeepSeaHybridClassifier...")
        
        # Define the base URL for your raw files on GitHub
        base_url = "https://github.com/ashish5507/SIH_AIgnition/raw/main/"
        
        # List of files to download
        files_to_ensure = {
            os.path.join(models_path, 'dna_cnn_full_v1.pth'): base_url + "models/dna_cnn_full_v1.pth",
            os.path.join(models_path, 'hashing_vectorizer_proto.joblib'): base_url + "models/hashing_vectorizer_proto.joblib",
            os.path.join(models_path, 'kmeans_clusterer_proto.joblib'): base_url + "models/kmeans_clusterer_proto.joblib",
            os.path.join(models_path, 'svd_transformer_proto.joblib'): base_url + "models/svd_transformer_proto.joblib",
            os.path.join(data_path, 'label_mappings_full.json'): base_url + "data/processed/label_mappings_full.json"
        }
        
        # Download each file
        for local_path, url in files_to_ensure.items():
            download_file_if_not_exists(url, local_path)

        # --- Load Models (same as before) ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ... (The rest of your __init__ and predict_batch methods remain exactly the same) ...
        self.dna_vocab = {'<pad>': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, '<unk>': 5}
        self.max_length = 500
        self.vectorizer = joblib.load(os.path.join(models_path, 'hashing_vectorizer_proto.joblib'))
        self.kmeans = joblib.load(os.path.join(models_path, 'kmeans_clusterer_proto.joblib'))
        self.svd = joblib.load(os.path.join(models_path, 'svd_transformer_proto.joblib'))
        with open(os.path.join(data_path, 'label_mappings_full.json'), 'r') as f:
            mappings = json.load(f)
        self.int_to_label = {int(k): v for k, v in mappings['int_to_label'].items()}
        VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES = len(self.dna_vocab), 64, len(self.int_to_label)
        self.cnn_model = DNA_CNN_Upgraded(VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES, self.max_length)
        self.cnn_model.load_state_dict(torch.load(os.path.join(models_path, 'dna_cnn_full_v1.pth'), map_location=self.device))
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
        print("INFO: All models downloaded and loaded successfully.")
    def _tokenize_batch(self, dna_sequences):
        token_list = []
        for seq in dna_sequences:
            tokenized_seq = [self.dna_vocab.get(base, self.dna_vocab['<unk>']) for base in seq.upper()]
            if len(tokenized_seq) > self.max_length: tokenized_seq = tokenized_seq[:self.max_length]
            else: tokenized_seq += [self.dna_vocab['<pad>']] * (self.max_length - len(tokenized_seq))
            token_list.append(tokenized_seq)
        return torch.LongTensor(token_list)

    def predict_batch(self, dna_sequences: list):
        if not dna_sequences: return []
        
        print("\n--- Starting New Batch Prediction (Adaptive Mode) ---")
        # --- 1. Get Top-N predictions for the entire batch ---
        tokenized_input = self._tokenize_batch(dna_sequences).to(self.device)
        with torch.no_grad():
            logits = self.cnn_model(tokenized_input)
            probabilities = F.softmax(logits, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
        
        # --- 2. Calculate ratios for the entire batch ---
        ratios = []
        for i in range(len(dna_sequences)):
            top_prob = top3_probs[i][0].item()
            third_prob = top3_probs[i][2].item()
            ratio = top_prob / (third_prob + 1e-9) # Add epsilon to avoid division by zero
            ratios.append(ratio)
        
        # --- 3. Compute the DYNAMIC threshold for this specific batch ---
        # A simple but effective method: mean + 1 standard deviation
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        dynamic_threshold = mean_ratio + std_ratio
        print(f"Dynamic Threshold for this batch: {dynamic_threshold:.2f} (Mean: {mean_ratio:.2f}, Std: {std_ratio:.2f})")

        # --- 4. Make final decisions ---
        final_results = []
        for i, seq in enumerate(dna_sequences):
            current_ratio = ratios[i]
            top_prob = top3_probs[i][0].item()
            top_label = self.int_to_label.get(top3_indices[i][0].item(), "Unknown")
            
            result = {
                "sequence_id": f"seq_{i+1}",
                "top_candidate": top_label,
                "confidence": top_prob,
                "decision_metric_ratio": current_ratio,
                "decision_threshold": dynamic_threshold,
            }

            if current_ratio > dynamic_threshold:
                result["final_label"] = top_label
                result["decision"] = "ACCEPTED"
            else:
                kmer_vector = self.vectorizer.transform([seq])
                reduced_vector = self.svd.transform(kmer_vector)
                cluster_id = self.kmeans.predict(reduced_vector)
                novel_label = f"Novel_OTU_{cluster_id[0]}"
                result["final_label"] = novel_label
                result["decision"] = "REJECTED"
            
            final_results.append(result)
        
        return final_results