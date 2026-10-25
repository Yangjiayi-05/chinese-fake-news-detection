import torch
import pickle
import os
import jieba
from config import PROCESSED_DIR, CHECKPOINT_DIR, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, MAX_LEN
from models.bilstm_attention import BiLSTM_Attention

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_vocab():
    with open(os.path.join(PROCESSED_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab['word2idx'])

    model = BiLSTM_Attention(vocab_size, EMBEDDING_DIM, HIDDEN_DIM,
                             output_dim=2, n_layers=N_LAYERS, dropout=DROPOUT)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_bilstm_attention_seed2024.pth'), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, vocab


def preprocess_text(text, word2idx, max_len=MAX_LEN):
    tokens = jieba.lcut(text)
    indices = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
    if len(indices) < max_len:
        indices += [word2idx['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor([indices], dtype=torch.long).to(DEVICE), tokens


def predict(text, model, vocab):
    word2idx = vocab['word2idx']
    tensor, tokens = preprocess_text(text, word2idx)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        conf = prob[0][pred].item()
        attn = model.get_attention(tensor).squeeze(0).cpu().numpy()

    label = '真实' if pred == 1 else '虚假'
    top_idx = attn.argsort()[-5:][::-1]
    top_words = [(tokens[i], round(float(attn[i]), 4)) for i in top_idx if i < len(tokens)]

    return label, conf, top_words


if __name__ == '__main__':
    model, vocab = load_model_and_vocab()
    print("Model and vocabulary loaded successfully. Please use app.py to input text.")
    # print("模型和词表加载成功，请通过app.py输入文本")
