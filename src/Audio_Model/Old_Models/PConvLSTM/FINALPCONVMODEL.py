import os
from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

from torch.utils.data import DataLoader, Dataset

from src.Audio_Model.TorchBiLSTM import get_dataloaders, split_data, collate_fn

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
    ORIGINAL_TRANSFORMED_ROOT = r"E:\CSC413_Data\ORDERED_DATA"
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
    test_count = num_actors - train_count - val_count

    train_actors = actors[:train_count]
    val_actors = actors[train_count:train_count+val_count]
    test_actors = actors[train_count+val_count:]

    # Identify class labels (the folder names inside the actor folders)
    sample_actor = train_actors[0] if len(train_actors) > 0 else actors[0]
    label_dirs = sorted([d for d in os.listdir(os.path.join(ORIGINAL_TRANSFORMED_ROOT, sample_actor))
                         if os.path.isdir(os.path.join(ORIGINAL_TRANSFORMED_ROOT, sample_actor, d))])
    label_to_idx = {label_name: i for i, label_name in enumerate(label_dirs)}

    train_pairs = get_file_label_pairs(train_actors, ORIGINAL_TRANSFORMED_ROOT, label_to_idx)
    val_pairs = get_file_label_pairs(val_actors, ORIGINAL_TRANSFORMED_ROOT, label_to_idx)
    test_pairs = get_file_label_pairs(test_actors, ORIGINAL_TRANSFORMED_ROOT, label_to_idx)

    # Save these splits
    torch.save(train_pairs, os.path.join(DATA_SPLIT_DIR, 'train.pt'))
    torch.save(val_pairs, os.path.join(DATA_SPLIT_DIR, 'val.pt'))
    torch.save(test_pairs, os.path.join(DATA_SPLIT_DIR, 'test.pt'))

    num_classes = len(label_to_idx)
    print("Data splits saved. Classes:", num_classes)


class MFCCDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fpath, label = self.pairs[idx]
        features = np.load(fpath)  # shape: (39, time) if mfcc+delta+delta-delta with n_mfcc=13
        features = torch.tensor(features, dtype=torch.float32).transpose(0,1) # (time, features)
        # label is an integer
        return features, label

def normalize_features(features):
    # Already normalized per file, but we can re-normalize if needed.
    # Here, do nothing or re-normalize if desired.
    # We'll leave it as is since we normalized during extraction.
    return features

def spec_augment(features, time_masking_ratio=0.2, freq_masking_ratio=0.2):
    # More aggressive than before
    B, T, F = features.size()
    time_mask_len = int(T * time_masking_ratio)
    freq_mask_len = int(F * freq_masking_ratio)

    for i in range(B):
        # Time masking
        if time_mask_len > 0:
            t0 = random.randint(0, max(T - time_mask_len, 0))
            features[i, t0:t0+time_mask_len, :] = 0.0
        # Frequency masking
        if freq_mask_len > 0:
            f0 = random.randint(0, max(F - freq_mask_len, 0))
            features[i, :, f0:f0+freq_mask_len] = 0.0
    return features

def collate_fn(batch, training=True):
    lengths = [item[0].size(0) for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)
    feature_dim = batch[0][0].size(1)
    inputs = torch.zeros(batch_size, max_len, feature_dim)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for i, (feat, label) in enumerate(batch):
        length = feat.size(0)
        inputs[i, :length, :] = feat
        targets[i] = label

    # More aggressive augmentation if training
    if training:
        inputs = spec_augment(inputs, time_masking_ratio=0.2, freq_masking_ratio=0.2)

    return inputs, lengths, targets

def get_dataloaders(batch_size=16):
    DATA_SPLIT_DIR = r"E:\CSC413_Data\Data_split"
    train_pairs = torch.load(os.path.join(DATA_SPLIT_DIR, 'train.pt'))
    val_pairs = torch.load(os.path.join(DATA_SPLIT_DIR, 'val.pt'))
    test_pairs = torch.load(os.path.join(DATA_SPLIT_DIR, 'test.pt'))

    train_dataset = MFCCDataset(train_pairs)
    val_dataset = MFCCDataset(val_pairs)
    test_dataset = MFCCDataset(test_pairs)

    # Create DataLoaders directly here
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, training=True))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, training=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, training=False))

    return train_loader, val_loader, test_loader




class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):  # reduced from 0.3 to 0.1
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, predictions, targets):
        with torch.no_grad():
            true_dist = torch.zeros_like(predictions)
            true_dist.fill_(self.smoothing / (predictions.size(1) - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(predictions, dim=1)
        return -(true_dist * log_probs).sum(dim=1).mean()

class StemBlock(nn.Module):
    def __init__(self, dropout=0.2):  # reduced dropout
        super(StemBlock, self).__init__()
        self.sconv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(3, 2)

        self.sconv2 = nn.Conv2d(32, 16, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, 3, 3, 1)
        self.bn3 = nn.BatchNorm2d(16)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.sconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.sconv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.dropout(x)
        return x

class PConvBlock(nn.Module):
    def __init__(self, in_channels=16, dropout=0.2):  # reduced dropout
        super(PConvBlock, self).__init__()
        self.sconv = nn.Conv2d(in_channels, 64, 3, 2, 1)
        self.sconv_bn = nn.BatchNorm2d(64)
        self.bconv = nn.Conv2d(in_channels, 32, 3, 2, 1)
        self.bconv_bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.match_channels = nn.Conv2d(48, 64, 1, 1)
        self.match_channels_bn = nn.BatchNorm2d(64)
        # Removed the dropout here to simplify
        # self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        s_out = self.sconv(x)
        s_out = self.sconv_bn(s_out)
        s_out = F.relu(s_out)

        b_out = self.bconv(x)
        b_out = self.bconv_bn(b_out)
        b_out = F.relu(b_out)

        p_out = self.pool(x)

        bp_cat = torch.cat([b_out, p_out], dim=1)
        bp_cat = self.match_channels(bp_cat)
        bp_cat = self.match_channels_bn(bp_cat)
        bp_cat = F.relu(bp_cat)

        out = s_out + bp_cat
        # removed dropout here
        out = F.relu(out)
        return out

def create_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = max(lengths)
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < max_len:
            mask[i, l:] = True
    return mask

class PConvBiLSTMAttentionModel(nn.Module):
    def __init__(self, num_classes=8, hidden_dim=64, num_layers=1, dropout=0.2):
        super(PConvBiLSTMAttentionModel, self).__init__()
        self.stem = StemBlock(dropout=0.2)
        self.pconv = PConvBlock(in_channels=16, dropout=0.2)
        # Reduced adaptive pooling from 40 to 30
        self.adaptive_pool = nn.AdaptiveAvgPool2d((30, 1))

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_input_size = 64

        self.bilstm = nn.LSTM(input_size=self.lstm_input_size,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=0.0)  # no dropout since num_layers=1

        self.dropout = nn.Dropout(dropout)

        attn_dim = 2 * hidden_dim
        self.query_proj = nn.Linear(attn_dim, attn_dim)
        self.key_proj = nn.Linear(attn_dim, attn_dim)
        self.value_proj = nn.Linear(attn_dim, attn_dim)

        self.fc_mid = nn.Linear(attn_dim, attn_dim)
        self.classifier = nn.Linear(attn_dim, num_classes)

    def forward(self, x, lengths):
        x = x.unsqueeze(1)  # (B,1,T,F)
        x = self.stem(x)    # (B,16,H',W')
        x = self.pconv(x)   # (B,64,H'',W'')

        x = self.adaptive_pool(x)  # (B,64,30,1)
        x = x.squeeze(-1)    # (B,64,30)
        x = x.permute(0, 2, 1) # (B,30,64)

        truncated_lengths = [min(l, 30) for l in lengths]
        packed = pack_padded_sequence(x, truncated_lengths, batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.bilstm(packed)
        padded_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        padded_out = self.dropout(padded_out)

        B, T, _ = padded_out.size()
        Q = self.query_proj(padded_out)
        K = self.key_proj(padded_out)
        V = self.value_proj(padded_out)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1)**0.5)

        mask = create_mask_from_lengths(truncated_lengths, max_len=T).to(padded_out.device)
        mask_expanded = mask.unsqueeze(1).expand(B, T, T)
        scores = scores.masked_fill(mask_expanded, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(mask_expanded, 0.0)

        context = torch.matmul(attn_weights, V)

        valid_lengths = torch.tensor(truncated_lengths, dtype=torch.float32, device=context.device).unsqueeze(1)
        context_sum = torch.sum(context * (~mask.unsqueeze(-1)).float(), dim=1)
        context_mean = context_sum / valid_lengths

        context_mean = self.fc_mid(context_mean)
        context_mean = F.relu(context_mean)
        context_mean = self.dropout(context_mean)
        logits = self.classifier(context_mean)
        return logits

#############################
# Training Loop with Early Stopping and Scheduler
#############################
def train_model(model, train_dataloader, val_dataloader, num_epochs=100, device='cuda', patience=10):
    model = model.to(device)
    # Reduced weight decay from 1e-2 to 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)  # reduced smoothing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_dataloader:
            inputs, lengths, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, dim=1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / len(train_dataloader.dataset)
        train_accuracy = 100.0 * total_correct / total_samples

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, lengths, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs, lengths)
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        val_accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Acc: {val_accuracy:.2f}%")

        scheduler.step()

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model saved at epoch {epoch+1} with val acc: {val_accuracy:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    return model


# The collate function would be defined outside of this code snippet,
# which pads sequences and returns (inputs, lengths, targets).
# The training function and model class are now ready.

if __name__ == '__main__':
    # split_data()
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders()
    train_model(PConvBiLSTMAttentionModel(), train_dataloader, val_dataloader)
