# -*- coding: utf-8 -*-
"""纯 BiLSTM 基线：不含注意力层，用于验证注意力机制的增量贡献。"""

import torch
import torch.nn as nn


class BiLSTM_NoAttention(nn.Module):
    """基线模型：取最后一个真实词位置的隐状态做句向量，不含注意力。"""

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

        # 真实长度用于定位最后一个有效词；clamp 防止整句全为补零时越界
        lengths = torch.clamp((text != 0).sum(dim=1), min=1) - 1
        context = lstm_out[torch.arange(lstm_out.size(0)), lengths]
        return self.fc(self.dropout(context))
