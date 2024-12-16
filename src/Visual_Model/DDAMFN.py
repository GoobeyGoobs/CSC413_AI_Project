import os
import sys

from sklearn.model_selection import KFold
from torch.cpu.amp import autocast
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import glob
import shutil
import random
from torch.cuda.amp import GradScaler

from DDA3D import DDAMNet
import torch.nn.functional as F

eps = sys.float_info.epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define paths
dataset_dir = r'D:\Uni\CSC413 Final Project\Datasets\TRANSFORMATIONS\V2_TRANSFORMED_DATA_16'  # Replace with your dataset directory path
train_dir = (r'D:\Uni\CSC413 Final Project\Datasets\DATA_SPLITS\V2_FPS16_AUGMENTS\TRAIN')       # Path to store training data
val_dir = r'D:\Uni\CSC413 Final Project\Datasets\DATA_SPLITS\V2_FPS16_AUGMENTS\VAL'           # Path to store validation data
test_dir = r'D:\Uni\CSC413 Final Project\Datasets\DATA_SPLITS\V2_FPS16_AUGMENTS\TEST'         # Path to store test data

# Function to copy files for each split
def split_dataset(actor_list, dest_dir):
    for actor in actor_list:
        actor_path = os.path.join(dataset_dir, actor)
        for class_folder in os.listdir(actor_path):
            class_path = os.path.join(actor_path, class_folder)
            if os.path.isdir(class_path):
                dest_class_path = os.path.join(dest_dir, class_folder)
                if not os.path.exists(dest_class_path):
                    os.makedirs(dest_class_path)
                # Copy all files in the class folder to the destination
                for file in os.listdir(class_path):
                    src_file = os.path.join(class_path, file)
                    dest_file = os.path.join(dest_class_path, f"{actor}_{file}")  # Keep track of actor in filename
                    shutil.copyfile(src_file, dest_file)

# class RAVDESSDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.file_paths = glob.glob(os.path.join(data_dir, '*', '*'))  # Get all files in all class folders
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         label = int(os.path.basename(os.path.dirname(file_path))) - 1  # Class labels are 1-8, convert to 0-7
#         tensor = torch.load(file_path)  # Load the tensor from file
#
#         if self.transform:
#             tensor = self.transform(tensor)
#
#         return tensor.float(), label

class RAVDESSDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.labels = []

        # Load data
        for label_folder in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_folder)
            if os.path.isdir(label_path) and label_folder.isdigit():  # Ensure it's a label folder
                label = int(label_folder) - 1  # Convert "01" to 0-based index for labels
                for video_file in os.listdir(label_path):
                    if video_file.startswith('Actor_') and video_file.endswith('.pt'):
                        video_path = os.path.join(label_path, video_file)
                        self.samples.append(video_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        frames = torch.load(video_path)  # Shape: (channels, frames, height, width)
        label = self.labels[idx]
        return frames, label


# Example of iterating through the DataLoader
# for data, labels in train_loader:
#     print(data.shape, labels)  # Data shape depends on the tensor size, labels are class indices
#     break




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='/data/affectnet/', help='AffectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=7, help='Number of class.')
    return parser.parse_args()


class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


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
    # args = parse_args()
    # root_dir = Path(__file__).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()

    model = DDAMNet(num_class=8, num_head=2)
    model.to(device)

    # # Get all actor directories
    # actors = [actor for actor in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, actor))]
    #
    # # Shuffle actors for random split
    # random.seed(42)  # For reproducibility
    # random.shuffle(actors)
    #
    # # Split actors into train, val, and test sets
    # train_actors = actors[:int(0.7 * len(actors))]
    # val_actors = actors[int(0.7 * len(actors)):int(0.85 * len(actors))]
    # test_actors = actors[int(0.85 * len(actors)):]
    #
    # # Split dataset into train, val, and test
    # split_dataset(train_actors, train_dir)
    # split_dataset(val_actors, val_dir)
    # split_dataset(test_actors, test_dir)

    # Create datasets
    dirname = os.path.dirname(__file__)
    print(dirname)
    print(dirname + r"/V2_FPS16_AUGMENTS/TRAIN")
    train_dataset = RAVDESSDataset(dirname + r"/V2_FPS16_AUGMENTS/TRAIN")
    val_dataset = RAVDESSDataset(dirname + r"/V2_FPS16_AUGMENTS/VAL")
    test_dataset = RAVDESSDataset(dirname + r"/V2_FPS16_AUGMENTS/TEST")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=6, pin_memory=True)


    # train_dataset = datasets.ImageFolder(f'{args.aff_path}/train')
    # if args.num_class == 7:  # ignore the 8-th class
    #     idx = [i for i in range(len(train_dataset)) if train_dataset.imgs[i][1] != 7]
    #     train_dataset = data.Subset(train_dataset, idx)
    #
    # print('Whole train set size:', train_dataset.__len__())
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=args.batch_size,
    #                                            num_workers=args.workers,
    #                                            sampler=ImbalancedDatasetSampler(train_dataset),
    #                                            shuffle=False,
    #                                            pin_memory=True)
    #
    # data_transforms_val = transforms.Compose([
    #     transforms.Resize((112, 112)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])])

    # val_dataset = datasets.ImageFolder(f'{args.aff_path}/val', transform=data_transforms_val)
    # if args.num_class == 7:  # ignore the 8-th class
    #     idx = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] != 7]
    #     val_dataset = data.Subset(val_dataset, idx)
    #
    # print('Validation set size:', val_dataset.__len__())
    #
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=args.batch_size,
    #                                          num_workers=args.workers,
    #                                          shuffle=False,
    #                                          pin_memory=True)
    best_acc = 0
    epochs = 40
    class_counts = np.bincount(train_dataset.labels)
    class_weights = 1. / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(class_weights).to(device)
    # criterion_cls = LabelSmoothingCrossEntropy(smoothing=0.1).to(device) TRY NEXT
    criterion_at = AttentionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2)
    # optimizer = torch.optim.Adam(params, lr=0.001,  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6) TRY NEXT

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

            # loss.backward()
            # optimizer.step()

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

            # elif acc > 0.632:
            #     torch.save({'iter': epoch,
            #                 'model_state_dict': model.state_dict(),
            #                 'optimizer_state_dict': optimizer.state_dict(), },
            #                os.path.join('checkpoints', "affecnet8_epoch" + str(epoch) + "_acc" + str(acc) + ".pth"))
            #     tqdm.write('Model saved.')

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
        bingo_cnt = 0
        sample_cnt = 0
        model.eval()
        print("Evaluating")
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            out, feat, heads = model(imgs)

            iter_cnt += 1
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)

        acc = bingo_cnt.float() / float(sample_cnt)
        acc = np.around(acc.numpy(), 4)
        print(f"Test accuracy:{acc}")

if __name__ == "__main__":
    # run_training()
    test_model()