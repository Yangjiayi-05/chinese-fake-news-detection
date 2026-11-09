# -*- coding: utf-8 -*-
"""核心模型：BiLSTM + 加性自注意力，use_mask 控制是否屏蔽补零。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_Attention(nn.Module):
    """双向 LSTM + 加性自注意力；use_mask 控制 softmax 前是否屏蔽 <PAD>。

    掩码做成开关而不是两份实现：全因子实验里要单独验证"有无掩码"，
    共用同一份代码才能保证除掩码外没有任何结构差异。
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, dropout=0.5, use_mask=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask
        # 用 -1e4 而不是 -1e9：混合精度(fp16)下 -1e9 会溢出成 -inf，softmax 会得到 NaN
        self.mask_value = -1e4

    def _score(self, text, lstm_out):
        """计算注意力打分；use_mask 为真时把补零位置压到极低分。"""
        scores = self.attention(lstm_out).squeeze(-1)
        if self.use_mask:
            return scores.masked_fill(text == 0, self.mask_value)
        return scores

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        # 屏蔽后归一化，PAD 位置权重趋近 0，句向量只由真实词决定
        weights = F.softmax(self._score(text, lstm_out), dim=1).unsqueeze(-1)
        context = torch.sum(lstm_out * weights, dim=1)
        return self.fc(self.dropout(context))

    def get_attention(self, text):
        """供可解释性可视化与系统高亮使用的注意力权重。"""
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        return F.softmax(self._score(text, lstm_out), dim=1)
