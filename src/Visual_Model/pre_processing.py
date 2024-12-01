# import os
# import random
#
# import numpy as np
# from PIL import Image
# from torch import Tensor
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import v2 as transforms
# from torchvision.transforms.functional import to_pil_image
#
# import torch
# import matplotlib.pyplot as plt
#
# # Config
# DATA_DIR = r"D:\Uni\CSC413 Final Project\Datasets\EXTRACTIONS"
# PROCESSED_DATA_DIR = r"D:\Uni\CSC413 Final Project\Datasets\TRANSFORMED_DATA_16"
# BATCH_SIZE = 2
# IMAGE_SIZE = (112, 112)
# NUM_WORKERS = 6
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# transform = transforms.Compose([
#     transforms.PILToTensor(),
#     transforms.Grayscale()
# ])
#
# # transforms.RandomErasing(scale= (0.01, 0.25)),
# #     transforms.RandomHorizontalFlip(),
# #     transforms.RandomAffine(degrees=(0, 360))
#
# def reduce_to_16_elements(tensor_list):
#     # Calculate indices to keep
#     original_length = len(tensor_list)
#     indices = np.linspace(0, original_length - 1, 16, dtype=int)
#
#     # Select only the elements at the calculated indices
#     reduced_list = [tensor_list[i] for i in indices]
#
#     return reduced_list
#
# def preprocess_and_save(data_dir, save_dir):
#     torch.set_printoptions(profile="full")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     for actor in os.listdir(data_dir):
#         actor_folder = os.path.join(data_dir, actor)
#         if os.path.isdir(actor_folder):
#             for extracted_dir in os.listdir(actor_folder):
#                 extracted_faces = os.path.join(actor_folder, extracted_dir)
#                 if os.path.isdir(extracted_faces):
#                     label = extracted_faces.split("-")[2]
#                     actor_save_folder = os.path.join(save_dir, "Actor_" + actor_folder.split("_")[1])
#                     if not os.path.exists(actor_save_folder):
#                         os.makedirs(actor_save_folder)
#                     label_save_folder = os.path.join(actor_save_folder, label)
#                     os.makedirs(label_save_folder, exist_ok=True)
#                     image_files = os.path.join(actor_folder, extracted_faces)
#                     save_path = os.path.join(label_save_folder, "video_" + str(len(os.listdir(label_save_folder))) + ".pt")
#                     video_tensor = []
#                     for image_file in os.listdir(image_files):
#                         if image_file.endswith(('.bmp')):
#                             image_path = os.path.join(image_files, image_file)
#
#                             image = Image.open(image_path)
#                             image_tensor = transform(image)
#                             video_tensor.append(image_tensor)
#                     video_tensor = reduce_to_16_elements(video_tensor)
#                     torch.save(torch.stack(video_tensor).permute(1, 0, 2, 3), save_path)
#                     for transform_num in range(3):
#                         save_path = os.path.join(label_save_folder,
#                                                  "video_" + str(len(os.listdir(label_save_folder))) + ".pt")
#                         video_tensor = []
#                         p = random.randint(0, 1)
#                         angle = random.randint(0, 360)
#                         translate = [0, 0]
#                         scale = 1
#                         for image_file in os.listdir(image_files):
#                             if image_file.endswith(('.bmp')):
#                                 image_path = os.path.join(image_files, image_file)
#
#                                 image = Image.open(image_path)
#                                 image_tensor = transform(image)
#                                 if p == 1:
#                                     if transform_num == 0:
#                                         image_tensor = transforms.functional.horizontal_flip_image(image_tensor)
#                                     elif transform_num == 1:
#                                         image_tensor = transforms.functional.affine_image(image_tensor, translate = translate, scale = scale, shear = [0.0, 0.0], angle = angle)
#                                     else:
#                                         image_tensor = transforms.functional.invert_image(image_tensor)
#                                     video_tensor.append(image_tensor)
#                         if video_tensor:
#                             video_tensor = reduce_to_16_elements(video_tensor)
#                             torch.save(torch.stack(video_tensor).permute(1, 0, 2, 3), save_path)
#                                 # img = to_pil_image(image_tensor)
#                                 # plt.imshow(img, cmap="gray")
#                                 # plt.show()

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA_DIR = r"D:\Uni\CSC413 Final Project\Datasets\EXTRACTIONS"
PROCESSED_DATA_DIR = r"D:\Uni\CSC413 Final Project\Datasets\TRANSFORMATIONS\V2_TRANSFORMED_DATA_16"
BATCH_SIZE = 2
IMAGE_SIZE = (112, 112)
NUM_WORKERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic transformation to convert PIL Image to tensor and resize
basic_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def get_augmentation_transform():
    return A.ReplayCompose([
        A.RandomResizedCrop(height=112, width=112, scale=(0.8, 1.0), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def reduce_to_16_elements(tensor_list):
    # Calculate indices to keep
    original_length = len(tensor_list)
    indices = np.linspace(0, original_length - 1, 16, dtype=int)

    # Select only the elements at the calculated indices
    reduced_list = [tensor_list[i] for i in indices]

    return reduced_list

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
                    image_files_dir = extracted_faces
                    # Original video without augmentation
                    save_path = os.path.join(label_save_folder, "video_" + str(len(os.listdir(label_save_folder))) + ".pt")
                    video_tensor = []
                    image_files = sorted(os.listdir(image_files_dir))
                    for image_file in image_files:
                        if image_file.endswith('.bmp'):
                            image_path = os.path.join(image_files_dir, image_file)
                            image = Image.open(image_path).convert('L')
                            image_np = np.expand_dims(np.array(image), axis=-1)  # Shape: (H, W, 1)
                            video_tensor.append(image_np)
                    video_tensor = reduce_to_16_elements(video_tensor)
                    # Convert to tensor and save
                    frames = []
                    basic_transform = A.Compose([
                        A.Resize(height=112, width=112),
                        A.Normalize(mean=(0.5,), std=(0.5,)),
                        ToTensorV2(),
                    ])
                    for frame in video_tensor:
                        transformed = basic_transform(image=frame)
                        frames.append(transformed['image'])
                    frames = torch.stack(frames).permute(1, 0, 2, 3)  # Shape: (channels, frames, height, width)
                    torch.save(frames, save_path)

                    # Generate augmented versions
                    num_augmented_versions = 9
                    for _ in range(num_augmented_versions):
                        save_path = os.path.join(label_save_folder,
                                                 "video_" + str(len(os.listdir(label_save_folder))) + ".pt")
                        video_tensor_augmented = []
                        # Get a random augmentation transform for the entire video
                        augmentation_transform = get_augmentation_transform()
                        # Apply to the first frame to get replay params
                        first_frame = video_tensor[0]
                        transformed = augmentation_transform(image=first_frame)
                        augmented_frame = transformed['image']
                        replay_params = transformed['replay']
                        video_tensor_augmented.append(augmented_frame)
                        # Apply the same augmentation to the rest of the frames using replay
                        for frame in video_tensor[1:]:
                            transformed = A.ReplayCompose.replay(replay_params, image=frame)
                            augmented_frame = transformed['image']
                            video_tensor_augmented.append(augmented_frame)
                        frames_augmented = torch.stack(video_tensor_augmented).permute(1, 0, 2, 3)
                        torch.save(frames_augmented, save_path)

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

    # torch.set_printoptions(profile="full")
    # train_loader = create_dataloaders(PROCESSED_DATA_DIR, BATCH_SIZE)
    # print("finished creating dataloaders")
    #
    # for images, labels in train_loader:
    #     for i in range(5):
    #         print(images[i].shape)  # Original shape of the image
    #         single_channel_image = images[i]  # Shape is [112, 112]
    #         for j in range(3):
    #             print(single_channel_image[:, j, :, :].shape)
    #             img = to_pil_image(single_channel_image[:, j, :, :])  # Convert to PIL image
    #             plt.imshow(img)
    #             plt.title(f"Label: {labels[i].item()}")
    #             plt.show()
    #
    #         # Convert to PIL image
    #
    #
    #     print(f"Batch images shape: {images.shape} on {DEVICE}")
    #     print(f"Batch labels: {labels}")
    #     break