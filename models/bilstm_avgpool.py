import torch
import torch.nn as nn

class BiLSTM_AvgPool(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(BiLSTM_AvgPool, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded) 
        
        # 1. 创建 Mask 矩阵 (batch_size, seq_len, 1)，真实词位置为 1.0，<PAD> 位置为 0.0
        mask = (text != 0).float().unsqueeze(-1)
        # 2. 把 <PAD> 位置的 LSTM 输出乘 0 屏蔽掉
        masked_lstm_out = lstm_out * mask 
        # 3. 只把真实词的输出加起来
        sum_out = torch.sum(masked_lstm_out, dim=1)
        # 4. 除以真实词的数量（注意分母不能为0）
        lengths = torch.clamp(mask.sum(dim=1), min=1e-9) 
        context = sum_out / lengths
        
        return self.fc(self.dropout(context))
