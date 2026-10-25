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
        return BiLSTM_NoAttention(vocab_size, EMBEDDING_DIM, HIDDEN_DIM,
                                  output_dim=2, n_layers=N_LAYERS, dropout=DROPOUT)
    else:
        return BiLSTM_Attention(vocab_size, EMBEDDING_DIM, HIDDEN_DIM,
                                output_dim=2, n_layers=N_LAYERS, dropout=DROPOUT)


def main():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_path = os.path.join(LOGS_DIR, f'log_{MODEL_NAME}_seed{SEED}.txt')
    log_file = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"数据集：MCFEND-中文平台合并")
    log(f"随机种子：SEED={SEED}")
    log(f"模型：{MODEL_NAME}")
    log(f"超参数：BATCH={BATCH_SIZE}, EMB={EMBEDDING_DIM}, HIDDEN={HIDDEN_DIM}, "
        f"DROPOUT={DROPOUT}, LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, "
        f"LABEL_SMOOTH={LABEL_SMOOTHING}, PATIENCE={PATIENCE}")
    log("=" * 80)

    with open(os.path.join(PROCESSED_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab['word2idx'])
    log(f"词汇表大小：{vocab_size}")

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = create_data_loader(os.path.join(PROCESSED_DIR, 'train.pkl'),
                                      BATCH_SIZE, shuffle=True, generator=g)
    dev_loader = create_data_loader(os.path.join(PROCESSED_DIR, 'dev.pkl'),
                                    BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(os.path.join(PROCESSED_DIR, 'test.pkl'),
                                     BATCH_SIZE, shuffle=False)

    model = build_model(MODEL_NAME, vocab_size).to(DEVICE)

    # ===== 处理类别不平衡：谣言(0)与事实(1) ≈ 2:1 =====
    class_weights = torch.tensor([1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    save_path = os.path.join(CHECKPOINT_DIR, f'best_{MODEL_NAME}_seed{SEED}.pth')

    best_f1 = 0.0
    counter = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        dev_loss, dev_acc, dev_prec, dev_rec, dev_f1, _, _ = evaluate(
            model, dev_loader, criterion, DEVICE)

        log(f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
            f"Dev Loss: {dev_loss:.4f} Dev Acc: {dev_acc:.4f} "
            f"Dev P: {dev_prec:.4f} R: {dev_rec:.4f} F1: {dev_f1:.4f}")

        if dev_f1 > best_f1 + MIN_DELTA:
            best_f1 = dev_f1
            torch.save(model.state_dict(), save_path)
            log(f"  -> 保存最佳模型（Dev F1={best_f1:.4f}）")
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                log(f"早停：连续 {PATIENCE} 轮 Dev F1 未提升")
                break

    log("=" * 80)
    log("加载最佳模型，在测试集上评估...")
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds = evaluate(
        model, test_loader, criterion, DEVICE)

    log("\n测试集结果：")
    log(f"Loss:      {test_loss:.4f}")
    log(f"Accuracy:  {test_acc:.4f}")
    log(f"Precision: {test_prec:.4f}")
    log(f"Recall:    {test_rec:.4f}")
    log(f"F1:        {test_f1:.4f}")

    np.save(os.path.join(CHECKPOINT_DIR, f'test_labels_{MODEL_NAME}_seed{SEED}.npy'),
            np.array(test_labels))
    np.save(os.path.join(CHECKPOINT_DIR, f'test_preds_{MODEL_NAME}_seed{SEED}.npy'),
            np.array(test_preds))
    log(f"\n已保存：test_labels / test_preds")

    log_file.close()


if __name__ == '__main__':
    main()