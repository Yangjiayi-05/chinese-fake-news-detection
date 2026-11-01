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

        # 1. 找出 <PAD> 的位置（是 0 的位置为 True）
        mask = (text == 0)
        # 2. 把 <PAD> 位置的得分变成负无穷大 (-1e9)，这样 Softmax 后它们就是 0！
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        return self.fc(self.dropout(context))

    # get_attention 方法顺便也加上 mask，方便你以后画热力图
    def get_attention(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        mask = (text == 0)
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        return attn_weights
