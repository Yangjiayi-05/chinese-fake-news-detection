# -*- coding: utf-8 -*-
"""TextCNN 基线：多尺度一维卷积 + 掩码最大池化。

与核心模型保持同一套"屏蔽补零"原则：
卷积前把补零位置的词向量置零，池化前把补零位置的响应压到 −1e4，
使其不会在最大池化中被选中。这样三个模型（词袋 / 卷积 / 循环）
的差异只来自结构本身。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim,
                 num_filters=100, kernel_sizes=(3, 4, 5),
                 dropout=0.5, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k, padding=(k - 1) // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.pad_value = -1e4

    def forward(self, text):
        mask = (text != 0).float()                 # [B, L]
        emb = self.embedding(text) * mask.unsqueeze(-1)   # 补零位向量置零
        x = emb.transpose(1, 2)                    # [B, E, L]

        feats = []
        for conv in self.convs:
            c = F.relu(conv(x))                    # [B, F, L_out]
            m = mask[:, :c.size(2)].unsqueeze(1)   # 对齐输出长度
            c = c.masked_fill(m == 0, self.pad_value)
            feats.append(c.max(dim=2).values)

        h = torch.cat(feats, dim=1)
        return self.fc(self.dropout(h))
