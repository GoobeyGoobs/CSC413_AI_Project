import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
import optuna
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.Audio_Model.Old_Models.wav2vec2 import EmbeddingDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def collate_fn(batch):
#     mfccs = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
#
#     # Pad them along the time dimension
#     mfccs_padded = pad_sequence(mfccs, batch_first=True)
#     labels = torch.stack(labels)
#     return mfccs_padded, labels

def collate_fn(batch):
    mfccs = [item[0] for item in batch]  # Now this is Wav2Vec2 embeddings
    labels = [item[1] for item in batch]

    # Pad them
    mfccs_padded = pad_sequence(mfccs, batch_first=True)
    labels = torch.stack(labels)

    return mfccs_padded, labels


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, lstm_dropout=0.2, fc_dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=lstm_dropout if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last time step
        out = lstm_out[:, -1, :]
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class MFCCDataset(Dataset):
    def __init__(self, pairs_list, global_mean, global_std):
        self.pairs = pairs_list
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fpath, label = self.pairs[idx]
        mfcc = np.load(fpath)  # shape (n_mfcc, time)
        mfcc = mfcc.T  # (time, n_mfcc)

        # Apply global normalization
        mfcc = (mfcc - self.global_mean) / (self.global_std + 1e-9)

        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return mfcc_tensor, label_tensor

def compute_global_mean_std(pairs_list):
    # Compute global mean and std over the entire training set
    all_features = []
    for fpath, _ in pairs_list:
        mfcc = np.load(fpath).T  # (time, n_mfcc)
        all_features.append(mfcc)
    all_features = np.concatenate(all_features, axis=0)  # (total_time, n_mfcc)
    global_mean = np.mean(all_features, axis=0)
    global_std = np.std(all_features, axis=0)
    return global_mean, global_std

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

def train_one_epoch(model, optimizer, criterion, train_loader, device, max_norm=5.0):
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

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def evaluate(model, criterion, val_loader, device):
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
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters (adjust as needed)
    hidden_dim = 32
    learning_rate = 1e-3
    num_layers = 1
    lstm_dropout = 0.2
    fc_dropout = 0.3
    batch_size = 16
    weight_decay = 1e-5

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    #input_dim = 48
    input_dim = 768
    num_classes = 8
    model = BiLSTMClassifier(input_dim, hidden_dim, num_classes,
                             num_layers=num_layers, lstm_dropout=lstm_dropout, fc_dropout=fc_dropout).to(device)

    # Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    num_epochs = 50

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Optuna pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Learning rate scheduling based on validation accuracy
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve_count = 0
            model_path = f"best_model_acc_{val_acc}.pt"
            torch.save(model.state_dict(), model_path)

    return best_val_acc

if __name__ == "__main__":
    split_data()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)  # adjust n_trials as needed

    print("Best trial:")
    trial = study.best_trial
    print(trial.params)
    print("Best validation accuracy:", trial.value)