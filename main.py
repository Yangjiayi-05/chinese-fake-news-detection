# -*- coding: utf-8 -*-
"""单次训练：一个模型配一个种子。

与旧版的关键区别是"选模型依据"——早停和最优权重都按验证集 Macro-F1 判定，
而不是正类 F1。数据不平衡时用正类 F1 选模型，会系统性偏向多数类。

用法：python main.py --model bilstm_attention --seed 42
"""
import argparse
import csv
import glob
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


def build_model(model_name, vocab_size, use_mask=True):
    if model_name == 'fasttext':
        return FastText(vocab_size, config.EMBEDDING_DIM,
                        output_dim=config.NUM_CLASSES, dropout=config.DROPOUT)

    kwargs = dict(output_dim=config.NUM_CLASSES, n_layers=config.N_LAYERS,
                  dropout=config.DROPOUT)
    if model_name == 'bilstm':
        return BiLSTM_NoAttention(vocab_size, config.EMBEDDING_DIM,
                                  config.HIDDEN_DIM, **kwargs)
    if model_name in ('bilstm_attention', 'bilstm_attention_nomask'):
        # 两个名字走同一份实现，只用 use_mask 区分；nomask 名字保留以兼容旧结果
        flag = use_mask and model_name != 'bilstm_attention_nomask'
        return BiLSTM_Attention(vocab_size, config.EMBEDDING_DIM,
                                config.HIDDEN_DIM, use_mask=flag, **kwargs)
    raise ValueError(f'未知模型：{model_name}')


def build_pretrained_embedding(vocab, train_x):
    """用训练集语料跑一个 Word2Vec，把词向量填进嵌入矩阵作为初始化。

    语料就是训练集自己（无监督，不碰测试集），所以不引入外部数据。
    首次运行约 20 秒，之后读 checkpoint 目录下的缓存。
    """
    vocab_size = len(vocab['word2idx'])
    dim = config.EMBEDDING_DIM
    cache = os.path.join(config.CHECKPOINT_DIR, f'w2v_init_{dim}_{vocab_size}.npy')
    if os.path.exists(cache):
        return np.load(cache)

    from gensim.models import Word2Vec
    idx2word = vocab['idx2word']
    # 用字符串词建句，并剔除补零：否则 75% 的 token 都是 <PAD>，会把词向量带偏
    sentences = []
    for seq in train_x:
        words = [idx2word[t] for t in seq if t != 0]
        if words:
            sentences.append(words)

    w2v = Word2Vec(sentences, vector_size=dim, window=config.W2V_WINDOW,
                   min_count=1, workers=1, epochs=config.W2V_EPOCHS,
                   seed=config.SPLIT_SEED)

    emb = np.random.normal(0, 0.1, (vocab_size, dim)).astype(np.float32)
    emb[vocab['word2idx']['<PAD>']] = 0.0
    hit = 0
    for idx, word in enumerate(idx2word):
        if word in w2v.wv:
            emb[idx] = w2v.wv[word]
            hit += 1
    print(f'      Word2Vec 初始化：{hit}/{vocab_size} 个词命中')
    np.save(cache, emb)
    return emb


def apply_reg(profile):
    """切换正则档位（A/B/C），只改 dropout / weight decay / word dropout 三项。"""
    cfg = config.REG_PROFILES[profile]
    config.DROPOUT = cfg['DROPOUT']
    config.WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    config.WORD_DROPOUT = cfg['WORD_DROPOUT']


def _signature(model_name, seed, reg, use_cw, use_w2v, use_mask):
    """把影响训练结果的全部设置打包，用于判断"这个配置是否已经跑过"。"""
    return {
        'model': model_name, 'seed': seed, 'reg': reg,
        'use_cw': bool(use_cw), 'use_w2v': bool(use_w2v),
        'use_mask': bool(use_mask), 'split': config.SPLIT_MODE,
        'max_len': config.MAX_LEN, 'epochs': config.EPOCHS,
        'patience': config.PATIENCE, 'dropout': config.DROPOUT,
        'weight_decay': config.WEIGHT_DECAY,
        'word_dropout': config.WORD_DROPOUT,
        'lr': config.LEARNING_RATE, 'batch': config.BATCH_SIZE,
        'emb': config.EMBEDDING_DIM, 'hidden': config.HIDDEN_DIM,
        'n_layers': config.N_LAYERS,
        'label_smoothing': config.LABEL_SMOOTHING,
    }


def _find_same_result(sig):
    """在已有结果里找配置完全一致的一次运行；找不到返回 None。"""
    for path in glob.glob(os.path.join(config.RESULTS_DIR, 'result_*.json')):
        try:
            with open(path, encoding='utf-8') as f:
                prev = json.load(f)
        except (OSError, ValueError):
            continue
        if prev.get('signature') == sig:
            return prev
    return None


def run(model_name, seed, reg='A', use_cw=None, use_w2v=True,
        use_mask=True, suffix='', force=False):
    """完整跑一次实验，返回指标字典（同时写入 results 目录）。

    reg      : 正则档位 A/B/C
    use_cw   : 是否使用类别加权交叉熵
    use_w2v  : 是否用 Word2Vec 初始化嵌入层
    use_mask : 注意力是否屏蔽 <PAD>（仅注意力模型有效）
    suffix   : 结果文件后缀（全因子变体用，如 _rC_c0_m0_v1）
    force    : 忽略"已存在"判断，强制重跑
    """
    set_seed(seed)
    apply_reg(reg)
    if use_cw is None:                       # 默认跟随最终配置（config.USE_CLASS_WEIGHT）
        use_cw = config.USE_CLASS_WEIGHT
    if model_name not in ('bilstm_attention', 'bilstm_attention_nomask'):
        use_mask = True                      # 两个基线没有注意力，掩码字段对其无意义
    tag = f'{model_name}_seed{seed}{suffix}'
    for _d in (config.CHECKPOINT_DIR, config.LOGS_DIR, config.RESULTS_DIR):
        config.ensure_dir(_d)

    # 全因子设计里同一个配置可能被多个批次引用，这里直接复用已有结果
    signature = _signature(model_name, seed, reg, use_cw, use_w2v, use_mask)
    if not force:
        prev = _find_same_result(signature)
        if prev is not None:
            if prev['tag'] == tag:
                print(f'[跳过] {tag}：该实验已完成，直接复用结果，不重复训练')
            else:
                print(f'[跳过] {tag}：模型、种子、正则、类权重、掩码、词向量与超参'
                      f'均与已完成的 {prev["tag"]} 完全一致，直接复用其结果')
            reused = dict(prev)
            reused['reused_as'] = tag
            return reused

    log_path = os.path.join(config.LOGS_DIR, f'log_{tag}.txt')

    print(f'\n===== {tag} | 正则={reg} | 类权重={use_cw} | 词向量={use_w2v} | 掩码={use_mask} | device={DEVICE} =====')
    with open(log_path, 'w', encoding='utf-8') as log_file:
        def log(msg):
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()

        log(f'模型 {model_name} | 种子 {seed} | 长度 {config.MAX_LEN} | 正则档位 {reg}')
        log(f'超参 EMB={config.EMBEDDING_DIM} HIDDEN={config.HIDDEN_DIM} '
            f'DROPOUT={config.DROPOUT} LR={config.LEARNING_RATE} WD={config.WEIGHT_DECAY} '
            f'WORD_DROPOUT={config.WORD_DROPOUT} | 类权重={use_cw} | 词向量={use_w2v}')

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

        model = build_model(model_name, len(vocab['word2idx']),
                            use_mask=use_mask).to(DEVICE)

        # 嵌入层初始化：随机 or Word2Vec 预训练（可训练，不冻结）
        if use_w2v and config.USE_PRETRAINED_EMB and hasattr(model, 'embedding'):
            with open(os.path.join(config.PROCESSED_DIR, 'train.pkl'), 'rb') as f:
                train_x = pickle.load(f)['X']
            emb = build_pretrained_embedding(vocab, train_x)
            model.embedding.weight.data.copy_(torch.from_numpy(emb))
            log(f'嵌入层用 Word2Vec 初始化（dim={config.EMBEDDING_DIM}，可训练）')

        # 类别权重：开启时按训练集分布计算，关闭时两类等权（消融用）
        n0, n1 = int(y_train.count(0)), int(y_train.count(1))
        total = n0 + n1
        if use_cw:
            w0, w1 = total / (2.0 * n0), total / (2.0 * n1)
        else:
            w0, w1 = 1.0, 1.0
        log(f'类别分布 谣言={n0} 事实={n1} | 权重 {w0:.4f}, {w1:.4f}')

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([w0, w1], dtype=torch.float).to(DEVICE),
            label_smoothing=config.LABEL_SMOOTHING)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY)

        save_path = os.path.join(config.CHECKPOINT_DIR, f'best_{tag}.pth')
        best_macro, best_epoch, counter = -1.0, 0, 0

        # 每轮指标另存一份 csv，画训练曲线时直接读表，不用从日志文本里抠
        curve_file = open(os.path.join(config.LOGS_DIR, f'curve_{tag}.csv'),
                          'w', newline='', encoding='utf-8')
        curve_writer = csv.writer(curve_file)
        curve_writer.writerow(['epoch', 'train_loss', 'train_acc', 'dev_loss',
                               'dev_acc', 'dev_f1', 'dev_macro_f1'])

        for epoch in range(1, config.EPOCHS + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, DEVICE,
                word_dropout=config.WORD_DROPOUT)
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
            curve_writer.writerow([epoch, f'{train_loss:.6f}', f'{train_acc:.6f}',
                                   f'{dev_loss:.6f}', f'{dev_acc:.6f}',
                                   f'{dev_f1:.6f}', f'{dev_macro:.6f}'])
            curve_file.flush()

            if counter >= config.PATIENCE:
                log(f'早停：验证集 Macro-F1 连续 {config.PATIENCE} 轮未提升')
                break

        curve_file.close()
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
            'tag': tag, 'model': model_name, 'seed': seed, 'reg': reg,
            'use_cw': use_cw, 'use_w2v': use_w2v, 'use_mask': use_mask,
            'signature': signature,
            'max_len': config.MAX_LEN, 'best_epoch': best_epoch,
            'dev_macro_f1': round(float(d_macro), 4),
            'test_acc': round(float(t_acc), 4),
            'test_precision': round(float(t_p), 4),
            'test_recall': round(float(t_r), 4),
            'test_f1': round(float(t_f1), 4),
            'test_macro_f1': round(float(t_macro), 4),
        }
        result_path = os.path.join(config.RESULTS_DIR, f'result_{tag}.json')
        if os.path.exists(result_path):
            with open(result_path, encoding='utf-8') as f:
                old = json.load(f)
            if old.get('signature') != signature:
                log(f'[覆盖] {tag} 已存在另一套配置的结果，本次训练将覆盖它')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=config.DEFAULT_MODEL,
                        choices=config.MODEL_LIST)
    parser.add_argument('--seed', type=int, default=config.SEEDS[0])
    parser.add_argument('--reg', default=config.REG, choices=list(config.REG_PROFILES))
    parser.add_argument('--no-class-weight', action='store_true',
                        help='关闭类别加权（消融用）')
    parser.add_argument('--no-w2v', action='store_true',
                        help='关闭 Word2Vec 初始化（消融用）')
    parser.add_argument('--no-mask', action='store_true',
                        help='关闭注意力掩码（消融用）')
    parser.add_argument('--suffix', default='', help='结果文件后缀')
    parser.add_argument('--force', action='store_true',
                        help='忽略"已跑过"判断，强制重跑')
    args = parser.parse_args()
    run(args.model, args.seed, reg=args.reg,
        use_cw=not args.no_class_weight,
        use_w2v=not args.no_w2v,
        use_mask=not args.no_mask,
        suffix=args.suffix,
        force=args.force)


if __name__ == '__main__':
    main()
