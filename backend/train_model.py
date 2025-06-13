# backend/train_model.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from backend.main import SimpleSignalNet, SIGNAL_LIBRARY_PATH

class SignalDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx]

def load_dataset():
    data = []
    labels = []
    metadata = []
    classes = sorted(os.listdir(SIGNAL_LIBRARY_PATH))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for c in classes:
        class_dir = os.path.join(SIGNAL_LIBRARY_PATH, c)
        for fname in os.listdir(class_dir):
            path = os.path.join(class_dir, fname)
            iq = np.fromfile(path, dtype=np.complex64)
            iq = np.abs(iq[:1024])
            if len(iq) < 1024:
                iq = np.pad(iq, (0, 1024 - len(iq)))
            data.append(iq)
            labels.append(class_to_idx[c])
            metadata.append({"filename": fname, "class": c, "path": path})

    return np.array(data), np.array(labels), classes, metadata

def train():
    data, labels, classes, metadata = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)
    train_loader = DataLoader(SignalDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(SignalDataset(X_val, y_val), batch_size=32)

    model = SimpleSignalNet(num_classes=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        acc = 0
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                acc += (output.argmax(1) == y).sum().item()
        print(f"Epoch {epoch}: Accuracy {acc / len(y_val):.2f}")

    torch.save({"model_state": model.state_dict(), "classes": classes}, "backend/signal_model.pth")

    # Сохраняем метаинформацию для фронтенда
    with open("backend/signal_metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    train()
