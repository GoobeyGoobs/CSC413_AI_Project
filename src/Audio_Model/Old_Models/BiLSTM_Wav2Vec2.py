import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from src.Audio_Model.Old_Models.wav2vec2 import EmbeddingDataset


def collate_fn(batch):
    mfccs = [item[0] for item in batch]  # Now this is Wav2Vec2 embeddings
    labels = [item[1] for item in batch]

    # Pad them
    mfccs_padded = pad_sequence(mfccs, batch_first=True)
    labels = torch.stack(labels)

    return mfccs_padded, labels


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.0):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

class MFCCDataset(Dataset):
    def __init__(self, pairs_list):
        self.pairs = pairs_list

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fpath, label = self.pairs[idx]
        mfcc = np.load(fpath)  # shape (n_mfcc, time)
        # Transpose to (time, n_mfcc) to be compatible with LSTM (batch_first=True)
        mfcc = mfcc.T  # (time, n_mfcc)

        # Convert to torch tensors
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return mfcc_tensor, label_tensor

def get_file_label_pairs(actors_list, root_dir, label_to_idx):
    pairs = []
    for actor in actors_list:
        actor_path = os.path.join(root_dir, actor)
        for lbl_dir in os.listdir(actor_path):
            lbl_path = os.path.join(actor_path, lbl_dir)
            if os.path.isdir(lbl_path):
                label_idx = label_to_idx[lbl_dir]
                for npy_file in os.listdir(lbl_path):
                    if npy_file.endswith('.npy'):
                        fpath = os.path.join(lbl_path, npy_file)
                        pairs.append((fpath, label_idx))
    return pairs

def split_data():
    # ORIGINAL_TRANSFORMED_ROOT = r"E:\CSC413_Data\ORDERED_DATA"
    ORIGINAL_TRANSFORMED_ROOT = r"E:\CSC413_Data\EMBEDDING_CLIPS"
    DATA_SPLIT_DIR = r"E:\CSC413_Data\Data_split"
    os.makedirs(DATA_SPLIT_DIR, exist_ok=True)

    # Set a random seed for reproducibility
    random.seed(42)

    actors = sorted([d for d in os.listdir(ORIGINAL_TRANSFORMED_ROOT) if os.path.isdir(os.path.join(ORIGINAL_TRANSFORMED_ROOT, d))])

    # Shuffle actors and split 80-10-10
    random.shuffle(actors)
    num_actors = len(actors)
    train_count = int(0.8 * num_actors)
    val_count = int(0.1 * num_actors)

    train_actors = actors[:train_count]
    val_actors = actors[train_count:train_count+val_count]
    test_actors = actors[train_count+val_count:]

    sample_actor = train_actors[0] if len(train_actors) > 0 else actors[0]
    label_dirs = sorted([d for d in os.listdir(os.path.join(ORIGINAL_TRANSFORMED_ROOT, sample_actor))
                         if os.path.isdir(os.path.join(ORIGINAL_TRANSFORMED_ROOT, sample_actor, d))])
    label_to_idx = {label_name: i for i, label_name in enumerate(label_dirs)}

    train_pairs = get_file_label_pairs(train_actors, ORIGINAL_TRANSFORMED_ROOT, label_to_idx)
    val_pairs = get_file_label_pairs(val_actors, ORIGINAL_TRANSFORMED_ROOT, label_to_idx)
    test_pairs = get_file_label_pairs(test_actors, ORIGINAL_TRANSFORMED_ROOT, label_to_idx)

    torch.save(train_pairs, os.path.join(DATA_SPLIT_DIR, 'train.pt'))
    torch.save(val_pairs, os.path.join(DATA_SPLIT_DIR, 'val.pt'))
    torch.save(test_pairs, os.path.join(DATA_SPLIT_DIR, 'test.pt'))

    num_classes = len(label_to_idx)
    print("Data splits saved. Classes:", num_classes)

def get_dataloaders(batch_size=16):
    data_split_dir = r"E:\CSC413_Data\Data_split"
    train_pairs = torch.load(os.path.join(data_split_dir, 'train.pt'))
    val_pairs = torch.load(os.path.join(data_split_dir, 'val.pt'))
    test_pairs = torch.load(os.path.join(data_split_dir, 'test.pt'))

    # Compute global mean/std from training set
    # global_mean, global_std = compute_global_mean_std(train_pairs)

    # train_dataset = MFCCDataset(train_pairs, global_mean, global_std)
    # val_dataset = MFCCDataset(val_pairs, global_mean, global_std)
    # test_dataset = MFCCDataset(test_pairs, global_mean, global_std)

    train_dataset = EmbeddingDataset(train_pairs)
    val_dataset = EmbeddingDataset(val_pairs)
    test_dataset = EmbeddingDataset(test_pairs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 768
    hidden_dim = 64
    learning_rate = 1e-3
    num_layers = 2
    dropout = 0.0
    batch_size = 16

    model = BiLSTMClassifier(input_dim, hidden_dim, 8, num_layers=num_layers, dropout=dropout).to(device)
    model = model.to(device)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-7, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val_acc = 0.0
    num_epochs = 50
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for (mfccs, labels) in train_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for (mfccs, labels) in val_loader:
                mfccs, labels = mfccs.to(device), labels.to(device)
                outputs = model(mfccs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
        val_acc = total_correct / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "../best_model.pth")
            print(f"Model saved at epoch {epoch+1} with val acc: {val_acc:.2f}%")

    return model

if __name__ == "__main__":
    split_data()
    train_model()