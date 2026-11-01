import torch
import torch.nn as nn

class BiLSTM_NoAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(BiLSTM_NoAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded) 
        
        # 1. 找到每个句子的真实长度（计算非0的个数）
        lengths = (text != 0).sum(dim=1)
        # 2. 防止某个句子全是0（极端情况），将最小长度设为1
        lengths = torch.clamp(lengths, min=1) - 1 
        # 3. 只提取每个句子最后一个真实词的输出
        batch_size = lstm_out.size(0)
        context = lstm_out[torch.arange(batch_size), lengths]
        
        return self.fc(self.dropout(context))
