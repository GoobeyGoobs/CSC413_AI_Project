import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, AdamW, BertModel
from torch.cuda.amp import autocast, GradScaler
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RAVDESSBertDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=64):
        self.df = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.emotion_to_label = {
            "Neutral": 0, "Calm": 1, "Happy": 2, "Sad": 3,
            "Angry": 4, "Fearful": 5, "Disgust": 6, "Surprised": 7
        }
        self.df['emotion'] = self.df['emotion'].map(self.emotion_to_label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        transcription = row['transcription']
        emotion_label = torch.tensor(row['emotion'], dtype=torch.long)

        encoding = self.tokenizer(
            transcription,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return input_ids, attention_mask, emotion_label

class AttentionHead(nn.Module):
    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor, attention_mask=None):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)
        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            scores = scores.float().masked_fill(attention_mask == 0, -1e9).to(input_tensor.dtype)
        attn = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(attn, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([AttentionHead(dim_inp, dim_out) for _ in range(num_heads)])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor, attention_mask):
        scores = torch.cat([head(input_tensor, attention_mask) for head in self.heads], dim=-1)
        return self.norm(self.linear(scores))

class Encoder(nn.Module):
    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor, attention_mask):
        context = self.attention(input_tensor, attention_mask)
        return self.norm(self.feed_forward(context))

class BERTCNN(nn.Module):
    def __init__(self, num_filters, kernel_sizes, dropout=0.1):
        super(BERTCNN, self).__init__()
        self.embedding = nn.Embedding(30522, 768)  
        self.encoder = Encoder(dim_inp=768, dim_out=768)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=768, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(len(kernel_sizes) * num_filters, 8) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        encoded = self.encoder(embedded, attention_mask) 

        encoded = encoded.permute(0, 2, 1) 
        cnn_outs = [torch.relu(conv(encoded)) for conv in self.convs]
        pooled = [torch.max(out, dim=2)[0] for out in cnn_outs]
        concat = torch.cat(pooled, dim=1)
        logits = self.fc(self.dropout(concat))
        return logits

class BertTrainer:
    def __init__(self, model, train_loader, val_loader, lr, epochs, model_save_path="model_weights"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.scaler = GradScaler()
        self.model_save_path = model_save_path 

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for input_ids, attention_mask, labels in self.train_loader:
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )
                with autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = total_loss / len(self.train_loader)
            epoch_accuracy = correct_predictions / total_samples
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            model_file = os.path.join(self.model_save_path, f"epoch_{epoch+1}_accuracy_{epoch_accuracy:.4f}.pt")
            torch.save(self.model.state_dict(), model_file)
            print(f"Model weights saved to: {model_file}")

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in self.val_loader:
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )
                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total


def objective(trial):
    learning_rate = trial.suggest_loguniform("lr", 1e-6, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_filters = trial.suggest_categorical("num_filters", [64, 128, 256])
    kernel_sizes = trial.suggest_categorical("kernel_sizes", [[2, 3, 4], [3, 4, 5]])
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    learning_rate = 0.00013136541947437186
    batch_size = 16
    num_filters = 64
    kernel_sizes = [3, 4, 5]
    dropout = 0.4441381671307012

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BERTCNN(
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
    ).to(device)

    trainer = BertTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=learning_rate,
        epochs=20,
        model_save_path=r"C:\Users\singh\PycharmProjects\CSC413_AI_Project\src\Text_Model\saved_models",
    )

    trainer.train()
    validation_accuracy = trainer.validate()
    return validation_accuracy

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset_path = r"C:\Users\singh\PycharmProjects\CSC413_AI_Project\src\Text_Model\BERTCNN_DATA.csv"
    dataset = RAVDESSBertDataset(file_path=dataset_path, tokenizer=tokenizer, max_len=64)

    learning_rate = 0.00013136541947437186
    batch_size = 16
    num_filters = 64
    kernel_sizes = [3, 4, 5]
    dropout = 0.4441381671307012

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = BERTCNN(
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(torch.load(r"C:\Users\singh\PycharmProjects\CSC413_AI_Project\src\Text_Model\saved_models\saved_bertcnn_model.pt"))

    test_accuracy = test_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")
