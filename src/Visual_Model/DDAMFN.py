import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import shutil

from DDA3D import DDAMNet
import torch.nn.functional as F

eps = sys.float_info.epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



dataset_dir = r'D:\Uni\CSC413 Final Project\Datasets\TRANSFORMATIONS\V2_TRANSFORMED_DATA_16'  
train_dir = (r'D:\Uni\CSC413 Final Project\Datasets\DATA_SPLITS\V2_FPS16_AUGMENTS\TRAIN')      
val_dir = r'D:\Uni\CSC413 Final Project\Datasets\DATA_SPLITS\V2_FPS16_AUGMENTS\VAL'           
test_dir = r'D:\Uni\CSC413 Final Project\Datasets\DATA_SPLITS\V2_FPS16_AUGMENTS\TEST'         


def split_dataset(actor_list, dest_dir):
    for actor in actor_list:
        actor_path = os.path.join(dataset_dir, actor)
        for class_folder in os.listdir(actor_path):
            class_path = os.path.join(actor_path, class_folder)
            if os.path.isdir(class_path):
                dest_class_path = os.path.join(dest_dir, class_folder)
                if not os.path.exists(dest_class_path):
                    os.makedirs(dest_class_path)
                
                for file in os.listdir(class_path):
                    src_file = os.path.join(class_path, file)
                    dest_file = os.path.join(dest_class_path, f"{actor}_{file}")  
                    shutil.copyfile(src_file, dest_file)

class RAVDESSDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.labels = []

        
        for label_folder in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_folder)
            if os.path.isdir(label_path) and label_folder.isdigit():  
                label = int(label_folder) - 1  
                for video_file in os.listdir(label_path):
                    if video_file.startswith('Actor_') and video_file.endswith('.pt'):
                        video_path = os.path.join(label_path, video_file)
                        self.samples.append(video_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        frames = torch.load(video_path)  
        label = self.labels[idx]
        return frames, label

class AttentionLoss(nn.Module):
    def __init__(self, ):
        super(AttentionLoss, self).__init__()

    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head - 1):
                for j in range(i + 1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt + 1
                    loss = loss + mse
            loss = cnt / (loss + eps)
        else:
            loss = 0
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)
        smoothed_labels = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smoothed_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        loss = -torch.sum(smoothed_labels * log_probs, dim=-1).mean()
        return loss

def run_training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()

    
    dirname = os.path.dirname(__file__)
    print(dirname)
    print(dirname + r"/V2_FPS16_AUGMENTS/TRAIN")
    train_dataset = RAVDESSDataset(dirname + r"/V2_FPS16_AUGMENTS/TRAIN")
    val_dataset = RAVDESSDataset(dirname + r"/V2_FPS16_AUGMENTS/VAL")

    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=6, pin_memory=True)

    model = DDAMNet(num_class=8, num_head=2, dropout=0.7)
    model.to(device)
    best_acc = 0
    epochs = 50
    class_counts = np.bincount(train_dataset.labels)
    class_weights = 1. / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(class_weights).to(device)
    criterion_at = AttentionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in tqdm(range(1, epochs + 1)):
        torch.cuda.empty_cache()
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        print("Epoch num: " + str(epoch))

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            print("Iteration: " + str(iter_cnt))
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, heads = model(imgs)
            loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (
        epoch, acc, running_loss, optimizer.param_groups[0]['lr']))
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (
        epoch, acc, running_loss, optimizer.param_groups[0]['lr']))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            print("Evaluating")
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                out, feat, heads = model(imgs)

                loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)

                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

            running_loss = running_loss / iter_cnt
            scheduler.step(running_loss)

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, acc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))

            if acc > 0.5:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join(r"C:\Users\singh\PycharmProjects\CSC413_AI_Project\src\Visual_Model\saved_models",
                                        "RAVDESS_epoch" + str(epoch) + "_acc" + str(acc) + ".pth"))
                tqdm.write('Model saved.')

def test_model():
    dirname = os.path.dirname(__file__)
    test_dataset = RAVDESSDataset(dirname + r"/V2_FPS16_AUGMENTS/TEST")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=6, pin_memory=True)

    model = DDAMNet(num_class=8, num_head=2).to(device)
    model.load_state_dict(torch.load(
        r"C:\Users\singh\PycharmProjects\CSC413_AI_Project\src\Visual_Model\saved_models\RAVDESS_epoch20_acc0.6294.pth",
        map_location=device)["model_state_dict"])

    with torch.no_grad():
        iter_cnt = 0
        correct_cnt = 0
        sample_cnt = 0
        model.eval()
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            out, feat, heads = model(imgs)

            iter_cnt += 1
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets)
            correct_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)

        acc = correct_cnt.float() / float(sample_cnt)
        acc = np.around(acc.numpy(), 4)
        print(f"Test accuracy:{acc}")

if __name__ == "__main__":
    run_training()
    # test_model()
