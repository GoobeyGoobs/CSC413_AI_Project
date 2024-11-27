import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import to_pil_image

import torch
import matplotlib.pyplot as plt

# Config
DATA_DIR = r"D:\Uni\CSC413 Final Project\Datasets\EXTRACTIONS"
PROCESSED_DATA_DIR = r"D:\Uni\CSC413 Final Project\Datasets\TRANSFORMED_DATA"
BATCH_SIZE = 256
IMAGE_SIZE = (112, 112)
NUM_WORKERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Grayscale(),
    transforms.RandomErasing(scale= (0.01, 0.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=(0, 360))
])

def preprocess_and_save(data_dir, save_dir):
    torch.set_printoptions(profile="full")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for actor in os.listdir(data_dir):
        actor_folder = os.path.join(data_dir, actor)
        if os.path.isdir(actor_folder):
            for extracted_dir in os.listdir(actor_folder):
                extracted_faces = os.path.join(actor_folder, extracted_dir)
                if os.path.isdir(extracted_faces):
                    label = extracted_faces.split("-")[2]
                    actor_save_folder = os.path.join(save_dir, "Actor_" + actor_folder.split("_")[1])
                    if not os.path.exists(actor_save_folder):
                        os.makedirs(actor_save_folder)
                    label_save_folder = os.path.join(actor_save_folder, label)
                    os.makedirs(label_save_folder, exist_ok=True)
                    image_files = os.path.join(actor_folder, extracted_faces)
                    for image_file in os.listdir(image_files):
                        if image_file.endswith(('.bmp')):
                            image_path = os.path.join(image_files, image_file)
                            save_path = os.path.join(label_save_folder, str(len(os.listdir(label_save_folder))) + ".pt")

                            image = Image.open(image_path)
                            image_tensor = transform(image)
                            # img = to_pil_image(image_tensor)
                            # plt.imshow(img, cmap="gray")
                            # plt.show()
                            torch.save(image_tensor, save_path)

class PreprocessedDataset(Dataset):
    def __init__(self, processed_data_dir):
        self.image_paths = []
        self.labels = []
        self.label_to_index = {}
        for actor in os.listdir(processed_data_dir):
            labels_dir = os.path.join(processed_data_dir, actor)
            for label_idx, label in enumerate(os.listdir(labels_dir)):
                label_folder = os.path.join(labels_dir, label)
                if os.path.isdir(label_folder):
                    self.label_to_index[label] = label_idx
                    for tensor_file in os.listdir(label_folder):
                        self.image_paths.append(os.path.join(label_folder, tensor_file))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_tensor = torch.load(self.image_paths[idx])
        label = self.labels[idx]
        return image_tensor, label

def create_dataloaders(data_dir, batch_size):
    dataset = PreprocessedDataset(data_dir)
    pin_memory = DEVICE.type == "cuda" and not any(
        torch.load(path).is_cuda for path in dataset.image_paths
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    return dataloader


if __name__ == "__main__":
    preprocess_and_save(DATA_DIR, PROCESSED_DATA_DIR)
    print("finished pre-processing")
    #
    # torch.set_printoptions(profile="full")
    # train_loader = create_dataloaders(PROCESSED_DATA_DIR, BATCH_SIZE)
    # print("finished creating dataloaders")
    #
    # for images, labels in train_loader:
    #     for i in range(5):
    #         img = to_pil_image(images[i])
    #         plt.imshow(img, cmap="gray")
    #         plt.title(f"Label: {labels[i].item()}")
    #         plt.show()
    #     print(f"Batch images shape: {images.shape} on {DEVICE}")
    #     print(f"Batch labels: {labels}")
    #     break