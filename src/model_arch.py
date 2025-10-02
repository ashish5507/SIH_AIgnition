# src/model_arch.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# This is the correct, final model architecture from your notebook.
class DNA_CNN_Upgraded(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, max_length):
        super(DNA_CNN_Upgraded, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=7, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.flatten = nn.Flatten()
        final_seq_length = int(max_length / 4 / 4)
        self.fc1 = nn.Linear(in_features=final_seq_length * 64, out_features=num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        logits = self.fc1(x)
        return logits