# -*- coding: utf-8 -*-
"""单次训练：一个模型配一个种子。

与旧版的关键区别是"选模型依据"——早停和最优权重都按验证集 Macro-F1 判定，
而不是正类 F1。数据不平衡时用正类 F1 选模型，会系统性偏向多数类。

用法：python main.py --model bilstm_attention --seed 42
"""
import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

import config
from models.bilstm import BiLSTM_NoAttention
from models.bilstm_attention import BiLSTM_Attention
from models.bilstm_avgpool import BiLSTM_AvgPool
from models.fasttext import FastText
from utils.helper import create_data_loader, evaluate, train_epoch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    """固定全部随机源，保证同一配置可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name, vocab_size):
    if model_name == 'fasttext':
        return FastText(vocab_size, config.EMBEDDING_DIM,
                        output_dim=config.NUM_CLASSES, dropout=config.DROPOUT)

    kwargs = dict(output_dim=config.NUM_CLASSES, n_layers=config.N_LAYERS,
                  dropout=config.DROPOUT)
    if model_name == 'bilstm':
        return BiLSTM_NoAttention(vocab_size, config.EMBEDDING_DIM,
                                  config.HIDDEN_DIM, **kwargs)
    if model_name == 'bilstm_avgpool':
        return BiLSTM_AvgPool(vocab_size, config.EMBEDDING_DIM,
                              config.HIDDEN_DIM, **kwargs)
    if model_name == 'bilstm_attention':
        return BiLSTM_Attention(vocab_size, config.EMBEDDING_DIM,
                                config.HIDDEN_DIM, **kwargs)
    raise ValueError(f'未知模型：{model_name}')


def run(model_name, seed):
    """完整跑一次实验，返回指标字典（同时写入 results 目录）。"""
    set_seed(seed)
    tag = f'{model_name}_seed{seed}'
    log_path = os.path.join(config.LOGS_DIR, f'log_{tag}.txt')

    print(f'\n===== {tag} | profile={config.PROFILE} | device={DEVICE} =====')
    with open(log_path, 'w', encoding='utf-8') as log_file:
        def log(msg):
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()

        log(f'模型 {model_name} | 种子 {seed} | 档位 {config.PROFILE} | 长度 {config.MAX_LEN}')
        log(f'超参 EMB={config.EMBEDDING_DIM} HIDDEN={config.HIDDEN_DIM} '
            f'DROPOUT={config.DROPOUT} LR={config.LEARNING_RATE} WD={config.WEIGHT_DECAY}')

        if not os.path.exists(os.path.join(config.PROCESSED_DIR, 'train.pkl')):
            raise FileNotFoundError(
                f'缺少预处理数据：{config.PROCESSED_DIR}，请先运行 utils/preprocess.py')

        with open(os.path.join(config.PROCESSED_DIR, 'vocab.pkl'), 'rb') as f:
            vocab = pickle.load(f)
        with open(os.path.join(config.PROCESSED_DIR, 'train.pkl'), 'rb') as f:
            y_train = pickle.load(f)['y']

        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = create_data_loader(
            os.path.join(config.PROCESSED_DIR, 'train.pkl'),
            config.BATCH_SIZE, shuffle=True, generator=g)
        dev_loader = create_data_loader(
            os.path.join(config.PROCESSED_DIR, 'dev.pkl'),
            config.BATCH_SIZE, shuffle=False)
        test_loader = create_data_loader(
            os.path.join(config.PROCESSED_DIR, 'test.pkl'),
            config.BATCH_SIZE, shuffle=False)

        model = build_model(model_name, len(vocab['word2idx'])).to(DEVICE)

        # 类别权重按训练集真实分布计算，不写死常数
        n0, n1 = int(y_train.count(0)), int(y_train.count(1))
        total = n0 + n1
        w0, w1 = total / (2.0 * n0), total / (2.0 * n1)
        log(f'类别分布 谣言={n0} 事实={n1} | 权重 {w0:.4f}, {w1:.4f}')

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([w0, w1], dtype=torch.float).to(DEVICE),
            label_smoothing=config.LABEL_SMOOTHING)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY)

        save_path = os.path.join(config.CHECKPOINT_DIR, f'best_{tag}.pth')
        best_macro, best_epoch, counter = -1.0, 0, 0

        for epoch in range(1, config.EPOCHS + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                                criterion, DEVICE)
            dev_loss, dev_acc, dev_p, dev_r, dev_f1, dev_macro, _, _ = evaluate(
                model, dev_loader, criterion, DEVICE)

            if dev_macro > best_macro + config.MIN_DELTA:
                best_macro, best_epoch, counter = dev_macro, epoch, 0
                torch.save(model.state_dict(), save_path)
                flag = '  <- best'
            else:
                counter += 1
                flag = ''

            log(f'Epoch {epoch:02d} | Train {train_loss:.4f}/{train_acc:.4f} | '
                f'Dev {dev_loss:.4f}/{dev_acc:.4f} P {dev_p:.4f} R {dev_r:.4f} '
                f'F1 {dev_f1:.4f} Macro-F1 {dev_macro:.4f}{flag}')
            if counter >= config.PATIENCE:
                log(f'早停：验证集 Macro-F1 连续 {config.PATIENCE} 轮未提升')
                break

        log(f'最优 epoch = {best_epoch}，验证集 Macro-F1 = {best_macro:.4f}')

        # 回到最优权重再导出概率，保证阈值搜索用的是最优模型的输出
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        _, d_acc, d_p, d_r, d_f1, d_macro, dev_y, _, dev_prob = evaluate(
            model, dev_loader, criterion, DEVICE, return_probs=True)
        _, t_acc, t_p, t_r, t_f1, t_macro, test_y, test_pred, test_prob = evaluate(
            model, test_loader, criterion, DEVICE, return_probs=True)

        log(f'验证集 Acc {d_acc:.4f} P {d_p:.4f} R {d_r:.4f} '
            f'F1 {d_f1:.4f} Macro-F1 {d_macro:.4f}')
        log(f'测试集 Acc {t_acc:.4f} P {t_p:.4f} R {t_r:.4f} '
            f'F1 {t_f1:.4f} Macro-F1 {t_macro:.4f}')
        log(classification_report(test_y, test_pred, digits=4))

        for name, arr in [
            ('test_labels', test_y), ('test_preds', test_pred), ('test_probs', test_prob),
            ('dev_labels', dev_y), ('dev_probs', dev_prob),
        ]:
            np.save(os.path.join(config.CHECKPOINT_DIR, f'{name}_{tag}.npy'),
                    np.array(arr))

        result = {
            'model': model_name, 'seed': seed, 'profile': config.PROFILE,
            'max_len': config.MAX_LEN, 'best_epoch': best_epoch,
            'dev_macro_f1': round(float(d_macro), 4),
            'test_acc': round(float(t_acc), 4),
            'test_precision': round(float(t_p), 4),
            'test_recall': round(float(t_r), 4),
            'test_f1': round(float(t_f1), 4),
            'test_macro_f1': round(float(t_macro), 4),
        }
        with open(os.path.join(config.RESULTS_DIR, f'result_{tag}.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=config.DEFAULT_MODEL,
                        choices=config.MODEL_LIST)
    parser.add_argument('--seed', type=int, default=config.SEEDS[0])
    args = parser.parse_args()
    run(args.model, args.seed)


if __name__ == '__main__':
    main()
