import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TextDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.X = data['X']
        self.y = data['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

def create_data_loader(data_path, batch_size, shuffle, generator=None):
    dataset = TextDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

def train_epoch(model, data_loader, optimizer, criterion, device, clip=5.0):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for texts, labels in data_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def evaluate(model, data_loader, criterion, device, return_probs=False):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 0].cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    if return_probs:
        return avg_loss, acc, precision, recall, f1, all_labels, all_preds, all_probs
    return avg_loss, acc, precision, recall, f1, all_labels, all_preds
