import os

import numpy as np
import optuna
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.Audio_Model.Old_Models.PConvLSTM.PConv import StemPlusPConvNet
from src.Audio_Model.TorchBiLSTM import BiLSTMClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CombinedModel(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=8):
        super(CombinedModel, self).__init__()
        self.vision_model = StemPlusPConvNet(in_channels=1).to(device)

        # We need to know the output shape of the vision model to set up the BiLSTM.
        # Let's do a dummy pass:
        dummy_input = torch.randn(1, 1, 100, 48).to(device)  # (batch, 1, time, mfcc_dim)
        with torch.no_grad():
            dummy_out = self.vision_model(dummy_input)
            # dummy_out: (1,16,H',W')
            _, c, Hp, Wp = dummy_out.shape
            # We'll treat H' as sequence length and C*W' as features:
            input_dim = c * Wp
            self.sequence_length = Hp

        self.lstm_model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    def forward(self, mfccs):
        # mfccs: (batch, max_time, 48)
        # Reshape to (batch, 1, max_time, 48) to feed into vision model
        mfccs = mfccs.unsqueeze(1)  # (batch, 1, max_time, 48)
        feat_map = self.vision_model(mfccs)  # (batch,16,H',W')

        b, c, H, W = feat_map.shape
        # Treat H as time, and c*W as features
        seq_data = feat_map.permute(0, 2, 1, 3) # (batch, H, c, W)
        seq_data = seq_data.reshape(b, H, c*W)  # (batch, H, 16*W)

        out = self.lstm_model(seq_data)  # (batch, num_classes)
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

def collate_fn(batch):
    # batch is a list of (mfcc_tensor, label_tensor)
    mfccs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # mfccs is a list of tensors of shape (time, n_mfcc)
    # Pad them along the time dimension so all have the same length
    mfccs_padded = pad_sequence(mfccs, batch_first=True)  # shape: (batch, max_time, n_mfcc)

    labels = torch.stack(labels)  # shape: (batch,)

    return mfccs_padded, labels

def get_dataloaders(batch_size=16):
    data_split_dir = r"E:\CSC413_Data\Data_split"
    train_pairs = torch.load(os.path.join(data_split_dir, 'train.pt'))
    val_pairs = torch.load(os.path.join(data_split_dir, 'val.pt'))
    test_pairs = torch.load(os.path.join(data_split_dir, 'test.pt'))

    train_dataset = MFCCDataset(train_pairs)
    val_dataset = MFCCDataset(val_pairs)
    test_dataset = MFCCDataset(test_pairs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

def train_one_epoch(model, optimizer, criterion, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def evaluate(model, criterion, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Hyperparameters to tune
    # hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128])
    # learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    # num_layers = trial.suggest_int('num_layers', 1, 2)
    # dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    # batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    hidden_dim = 128
    learning_rate = 1e-4
    num_layers = 2
    dropout = 0.0
    batch_size = 32

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # num_classes and input_dim must be known from the data
    # We have num_classes from above and we know n_mfcc=13 (from original instructions)
    input_dim = 48
    model = CombinedModel(hidden_dim=hidden_dim, num_classes=8).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-7, weight_decay=0.0)
    best_val_acc = 0.0
    num_epochs = 1000  # You can increase this
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_acc = evaluate(model, criterion, val_loader)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
        # Pruning if no improvement
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # If val_acc > 0.6, save the model
            if val_acc > 0.6:
                model_path = f"best_model_trial_{trial.number}.pt"
                torch.save(model.state_dict(), model_path)

    return best_val_acc

if __name__ == "__main__":
    # split_data()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)  # adjust n_trials as needed

    print("Best trial:")
    trial = study.best_trial
    print(trial.params)
    print("Best validation accuracy:", trial.value)