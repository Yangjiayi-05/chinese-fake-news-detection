import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random
import numpy as np
from config import *
from utils.helper import create_data_loader, train_epoch, evaluate
from models.bilstm_attention import BiLSTM_Attention
from models.bilstm import BiLSTM_NoAttention
from models.bilstm_avgpool import BiLSTM_AvgPool
from models.fasttext import FastText

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(model_name, vocab_size):
    if model_name == 'fasttext':
        return FastText(vocab_size, EMBEDDING_DIM, output_dim=2, dropout=DROPOUT)
    elif model_name == 'bilstm':
        return BiLSTM_NoAttention(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim=2, n_layers=N_LAYERS, dropout=DROPOUT)
    elif model_name == 'bilstm_avgpool':
        return BiLSTM_AvgPool(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim=2, n_layers=N_LAYERS, dropout=DROPOUT)
    else:
        return BiLSTM_Attention(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim=2, n_layers=N_LAYERS, dropout=DROPOUT)

def main():
    set_seed(SEED)
    
    log_path = os.path.join(LOGS_DIR, f'log_{MODEL_NAME}_seed{SEED}.txt')
    log_file = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Model: {MODEL_NAME} | Seed: {SEED} | Device: {DEVICE}")
    
    with open(os.path.join(PROCESSED_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab['word2idx'])

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = create_data_loader(os.path.join(PROCESSED_DIR, 'train.pkl'), BATCH_SIZE, shuffle=True, generator=g)
    dev_loader = create_data_loader(os.path.join(PROCESSED_DIR, 'dev.pkl'), BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(os.path.join(PROCESSED_DIR, 'test.pkl'), BATCH_SIZE, shuffle=False)

    model = build_model(MODEL_NAME, vocab_size).to(DEVICE)
    class_weights = torch.tensor([1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    save_path = os.path.join(CHECKPOINT_DIR, f'best_{MODEL_NAME}_seed{SEED}.pth')

    best_f1 = 0.0
    counter = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        dev_loss, dev_acc, dev_prec, dev_rec, dev_f1, _, _ = evaluate(model, dev_loader, criterion, DEVICE)

        # ===== 这里恢复了详细的日志打印 =====
        log(f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
            f"Dev Loss: {dev_loss:.4f} Dev Acc: {dev_acc:.4f} "
            f"Dev P: {dev_prec:.4f} R: {dev_rec:.4f} F1: {dev_f1:.4f}")

        if dev_f1 > best_f1 + MIN_DELTA:
            best_f1 = dev_f1
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                log("Early stopping triggered.")
                break

    # Evaluate on Test Set
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion, DEVICE, return_probs=True)

    log(f"Test Results | Acc: {test_acc:.4f} | P: {test_prec:.4f} | R: {test_rec:.4f} | F1: {test_f1:.4f}")
    
    # Save outputs for visualization and threshold optimization
    np.save(os.path.join(CHECKPOINT_DIR, f'test_labels_{MODEL_NAME}_seed{SEED}.npy'), np.array(test_labels))
    np.save(os.path.join(CHECKPOINT_DIR, f'test_preds_{MODEL_NAME}_seed{SEED}.npy'), np.array(test_preds))
    np.save(os.path.join(CHECKPOINT_DIR, f'test_probs_{MODEL_NAME}_seed{SEED}.npy'), np.array(test_probs))
    
    log_file.close()

if __name__ == '__main__':
    main()
