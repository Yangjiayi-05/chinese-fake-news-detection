import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_Attention_NoMask(nn.Module):
    """消融模型：结构与主模型完全一致，只去掉对 <PAD> 的屏蔽。

    两者唯一差别是 softmax 前有没有 masked_fill，因此这一组的差值
    可以直接归因于 Padding Mask 本身，而不是结构差异。
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _forward_features(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        scores = self.attention(lstm_out).squeeze(-1)
        # 注意：这里不做 masked_fill，PAD 位置会一起参与 softmax
        weights = F.softmax(scores, dim=1)
        return lstm_out, weights

    def forward(self, text):
        lstm_out, weights = self._forward_features(text)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return self.fc(self.dropout(context))

    def get_attention(self, text):
        return self._forward_features(text)[1]
