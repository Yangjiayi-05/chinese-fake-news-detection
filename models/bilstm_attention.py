import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.fc(self.dropout(context))

    def get_attention(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights