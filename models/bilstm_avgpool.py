import torch
import torch.nn as nn


class BiLSTM_AvgPool(nn.Module):
    """消融模型：用掩码平均池化替代注意力，其余结构与主模型一致。"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)

        # 真实词位置为 1、补零位置为 0，求和再除以真实词数即得平均池化
        mask = (text != 0).float().unsqueeze(-1)
        summed = torch.sum(lstm_out * mask, dim=1)
        lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
        context = summed / lengths
        return self.fc(self.dropout(context))
