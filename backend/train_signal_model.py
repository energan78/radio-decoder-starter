import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import h5py
from backend.signal_utils import load_signal

SIGNAL_LIBRARY_PATH = "backend/signal_library"
RADIOML2018_PATH = "backend/radioml2018/RML2018.01A.h5"
MODEL_PATH = "backend/signal_model.pth"
SAMPLE_LEN = 1024

class SignalDataset(Dataset):
    def __init__(self, root_dir, sample_len=1024, radioml2018_path=None):
        self.samples = []
        self.labels = []
        self.classes = []
        self.class_to_idx = {}

        # 1. Собственные данные
        if os.path.exists(root_dir):
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            for label in self.classes:
                class_dir = os.path.join(root_dir, label)
                for fname in os.listdir(class_dir):
                    fpath = os.path.join(class_dir, fname)
                    self.samples.append((fpath, self.class_to_idx[label]))

        # 2. Данные RadioML 2018.01A
        if radioml2018_path and os.path.exists(radioml2018_path):
            with h5py.File(radioml2018_path, 'r') as f:
                X = f['X']  # Только ссылка на данные, не загружает всё в память
                Y = f['Y']
                mods = [f['classes'][i].decode() for i in range(f['classes'].shape[0])] if 'classes' in f else [str(i) for i in range(Y.shape[1])]
                for i in range(X.shape[0]):
                    mod_idx = np.argmax(Y[i])
                    mod = mods[mod_idx]
                    if mod not in self.class_to_idx:
                        self.class_to_idx[mod] = len(self.class_to_idx)
                        self.classes.append(mod)
                    iq = X[i, 0] + 1j * X[i, 1]
                    self.samples.append((iq, self.class_to_idx[mod]))

        self.sample_len = sample_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        if isinstance(sample, str):
            data = load_signal(sample)
        else:
            data = sample
        data = np.abs(data)
        if len(data) < self.sample_len:
            data = np.pad(data, (0, self.sample_len - len(data)), 'constant')
        else:
            data = data[:self.sample_len]
        data = data.astype(np.float32)
        return torch.tensor(data), label

class SimpleSignalNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(SAMPLE_LEN, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvSignalNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * (SAMPLE_LEN // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, SAMPLE_LEN)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    dataset = SignalDataset(SIGNAL_LIBRARY_PATH, SAMPLE_LEN, radioml2018_path=RADIOML2018_PATH)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = ConvSignalNet(num_classes=len(dataset.classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"Epoch {epoch+1}: val accuracy = {correct/total:.2f}")

    torch.save({
        "model_state": model.state_dict(),
        "classes": dataset.classes
    }, MODEL_PATH)
    print("Модель сохранена:", MODEL_PATH)

if __name__ == "__main__":
    train()

