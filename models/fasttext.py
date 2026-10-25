import torch
import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout=0.5):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        mask = (text != 0).float().unsqueeze(-1)
        pooled = (embedded * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.dropout(pooled))