# -*- coding: utf-8 -*-
"""训练与评估的通用工具：随机种子、指标计算、早停判据。"""

import pickle

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """把预处理好的 {'X': [[idx...]], 'y': [...]} 包装成 Dataset。"""

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.X = data['X']
        self.y = data['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.long),
                torch.tensor(self.y[idx], dtype=torch.long))


def create_data_loader(data_path, batch_size, shuffle, generator=None):
    dataset = TextDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator)


def train_epoch(model, data_loader, optimizer, criterion, device, clip=5.0,
                word_dropout=0.0):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for texts, labels in data_loader:
        texts, labels = texts.to(device), labels.to(device)
        # Word Dropout：按概率把真实词替换成 <UNK>(1)，不动补零位(0)
        if word_dropout > 0:
            drop = torch.rand(texts.shape, device=device) < word_dropout
            texts = torch.where(drop & (texts != 0),
                                torch.ones_like(texts), texts)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        # 梯度裁剪，防止 LSTM 在长序列上梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(data_loader), accuracy_score(all_labels, all_preds)


def evaluate(model, data_loader, criterion, device, return_probs=False):
    """返回 (loss, acc, precision, recall, f1, macro_f1, labels, preds[, probs])。

    macro_f1 才是论文的主指标；f1 是正类（事实）的二分类 F1，仅作参考。
    probs 是判定为"事实"的概率，用于后续三档预警的阈值搜索。
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            total_loss += criterion(outputs, labels).item()

            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary')
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro')

    if return_probs:
        return (avg_loss, acc, precision, recall, f1, macro_f1,
                all_labels, all_preds, all_probs)
    return avg_loss, acc, precision, recall, f1, macro_f1, all_labels, all_preds
