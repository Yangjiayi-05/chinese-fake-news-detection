# -*- coding: utf-8 -*-
"""推理：加载部署权重，输出"事实概率"和注意力关键词。

两处与旧版不同：
1. 序列长度统一取 config.MAX_LEN，避免训练 256、推理 200 的位置错位；
2. 只返回概率，不在这里做标签判定，标签由 app.py 用统一阈值决定。
"""
import os
import pickle

import jieba
import torch

import config
from models.bilstm_attention import BiLSTM_Attention

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_vocab(model_name=None, seed=None):
    """加载词表与权重；文件缺失时直接报出期望路径，省得在云上翻日志。"""
    model_name = model_name or config.DEFAULT_MODEL
    seed = config.DEPLOY_SEED if seed is None else seed
    if model_name != 'bilstm_attention':
        raise ValueError('应用系统只部署 BiLSTM+Attention 模型')

    vocab_path = os.path.join(config.PROCESSED_DIR, 'vocab.pkl')
    ckpt_path = os.path.join(config.CHECKPOINT_DIR,
                             f'best_{model_name}_seed{seed}.pth')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f'缺少词表：{vocab_path}，请先运行 utils/preprocess.py')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'缺少权重：{ckpt_path}，请先运行 main.py')

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    model = BiLSTM_Attention(len(vocab['word2idx']), config.EMBEDDING_DIM,
                             config.HIDDEN_DIM, output_dim=config.NUM_CLASSES,
                             n_layers=config.N_LAYERS, dropout=config.DROPOUT)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, vocab


def preprocess_text(text, word2idx, max_len=None):
    """分词、截断、补零；tokens 同步截断，保证与注意力下标一一对应。"""
    max_len = config.MAX_LEN if max_len is None else max_len
    tokens = jieba.lcut(text)[:max_len]
    indices = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
    indices += [0] * (max_len - len(indices))
    return torch.tensor([indices], dtype=torch.long).to(DEVICE), tokens


def predict(text, model, vocab, top_k=8):
    """返回 (事实概率, 关键词及权重, 分词结果)。"""
    tensor, tokens = preprocess_text(text, vocab['word2idx'])
    with torch.no_grad():
        logits = model(tensor)
        p_real = float(torch.softmax(logits, dim=1)[0, 1].item())
        attn = model.get_attention(tensor)[0, :len(tokens)].cpu().numpy()

    top_words = []
    for i in attn.argsort()[::-1]:
        if len(tokens[i].strip()) < 1:
            continue
        top_words.append((tokens[i], round(float(attn[i]), 4)))
        if len(top_words) >= top_k:
            break
    return p_real, top_words, tokens
