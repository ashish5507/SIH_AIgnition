# src/predictor_proto.py (for Simple Method - Option A)

import torch
import torch.nn.functional as F
import json
import joblib
import os
import numpy as np
# The 'requests' import is no longer needed
from src.model_arch import DNA_CNN_Upgraded

class DeepSeaHybridClassifier:
    def __init__(self):
        print("INFO: Initializing DeepSeaHybridClassifier with QUANTIZED model...")

        # These are now relative paths, as the files are copied with the app
        self.models_path = 'models/'
        self.data_path = 'data/processed/'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # NO DOWNLOAD LOGIC NEEDED

        self.cnn_model = None
        self.vectorizer = None
        self.kmeans = None
        self.svd = None
        self.int_to_label = None

        self.dna_vocab = {'<pad>': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, '<unk>': 5}
        self.max_length = 500
        print("INFO: Initialization complete. Quantized model will be loaded from local files on first use.")

    def _lazy_load_models(self):
        """Load quantized CNN + clustering models only once."""
        if self.cnn_model is None:
            print("INFO: Loading models into memory from local files...")

            self.vectorizer = joblib.load(os.path.join(self.models_path, 'hashing_vectorizer_proto.joblib'))
            self.kmeans = joblib.load(os.path.join(self.models_path, 'kmeans_clusterer_proto.joblib'))
            self.svd = joblib.load(os.path.join(self.models_path, 'svd_transformer_proto.joblib'))

            with open(os.path.join(self.data_path, 'label_mappings_full.json'), 'r') as f:
                mappings = json.load(f)
            self.int_to_label = {int(k): v for k, v in mappings['int_to_label'].items()}

            # --- How the quantized model is loaded ---
            # 1. Create the original model structure with the CORRECT number of classes.
            VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES, MAX_LENGTH = len(self.dna_vocab), 64, 63582, self.max_length
            model_fp32 = DNA_CNN_Upgraded(VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES, MAX_LENGTH)
            
            # 2. Prepare it for quantization.
            model_quantized = torch.quantization.quantize_dynamic(
                model_fp32, {torch.nn.Conv1d, torch.nn.Linear}, dtype=torch.qint8
            )

            # 3. Load the saved weights into the quantized architecture.
            model_path = os.path.join(self.models_path, 'dna_cnn_quantized_v1.pth')
            model_quantized.load_state_dict(torch.load(model_path, map_location=self.device))
            
            self.cnn_model = model_quantized
            self.cnn_model.to(self.device)
            self.cnn_model.eval()

            print("INFO: All models loaded successfully (lazy load complete).")

    # --- NO CHANGES to the methods below this line ---
    def _tokenize_batch(self, dna_sequences):
        # ... (code is identical)
        token_list = []
        for seq in dna_sequences:
            tokenized_seq = [self.dna_vocab.get(base, self.dna_vocab['<unk>']) for base in seq.upper()]
            if len(tokenized_seq) > self.max_length:
                tokenized_seq = tokenized_seq[:self.max_length]
            else:
                tokenized_seq += [self.dna_vocab['<pad>']] * (self.max_length - len(tokenized_seq))
            token_list.append(tokenized_seq)
        return torch.LongTensor(token_list)

    def predict_batch(self, dna_sequences: list):
        # ... (code is identical)
        if not dna_sequences:
            return []
        self._lazy_load_models()
        print("\n--- Starting New Batch Prediction (Adaptive Mode) ---")
        # (rest of your predict_batch logic)
        tokenized_input = self._tokenize_batch(dna_sequences).to(self.device)
        with torch.no_grad():
            logits = self.cnn_model(tokenized_input)
            probabilities = F.softmax(logits, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
        ratios = []
        for i in range(len(dna_sequences)):
            top_prob = top3_probs[i][0].item()
            third_prob = top3_probs[i][2].item()
            ratio = top_prob / (third_prob + 1e-9)
            ratios.append(ratio)
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        dynamic_threshold = mean_ratio + std_ratio
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