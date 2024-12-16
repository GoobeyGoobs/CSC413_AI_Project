import os
import random

import librosa
import torch
from six import text_type
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import BertTokenizer

VISUAL_DIR = r"D:\Uni\CSC413 Final Project\Datasets\EXTRACTIONS"
AUDIO_DIR = r"E:\CSC413_Data\RAW_CLIPS"
TEXT_DIR = r"E:\CSC413_Data\EXTRACTED_TEXT"
PROCESSED_DATA_DIR = r"E:\CSC413_Data\FUSION_DATA"
IMAGE_SIZE = (112, 112)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SAMPLE_RATE = 22050
N_MFCC = 48
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
FMIN = 50
FMAX = 8000


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


basic_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def reduce_to_16_elements(tensor_list):
    original_length = len(tensor_list)
    indices = np.linspace(0, original_length - 1, 16, dtype=int)

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
                    combined_data = []
                    label = extracted_faces.split("-")[2]
                    curr_actor = "Actor_" + actor_folder.split("_")[1]


                    actor_save_folder = os.path.join(save_dir, curr_actor)
                    if not os.path.exists(actor_save_folder):
                        os.makedirs(actor_save_folder)
                    label_save_folder = os.path.join(actor_save_folder, label)
                    os.makedirs(label_save_folder, exist_ok=True)
                    image_files_dir = extracted_faces

                    save_path = os.path.join(label_save_folder,
                                             "combined" + str(len(os.listdir(label_save_folder))) + ".pt"
                                             )
                    
                    video_tensor = []
                    image_files = sorted(os.listdir(image_files_dir))
                    for image_file in image_files:
                        if image_file.endswith('.bmp'):
                            image_path = os.path.join(image_files_dir, image_file)
                            image = Image.open(image_path).convert('L')
                            image_np = np.expand_dims(np.array(image), axis=-1)  
                            video_tensor.append(image_np)
                    video_tensor = reduce_to_16_elements(video_tensor)
                    
                    frames = []
                    basic_transform = A.Compose([
                        A.Resize(height=112, width=112),
                        A.Normalize(mean=(0.5,), std=(0.5,)),
                        ToTensorV2(),
                    ])
                    for frame in video_tensor:
                        transformed = basic_transform(image=frame)
                        frames.append(transformed['image'])
                    frames = torch.stack(frames).permute(1, 0, 2, 3)  
                    combined_data.append(frames)

                    
                    audio_filename = extracted_dir.split("_")[0] + ".wav"
                    audio_filename = "03" + audio_filename[2:]
                    audio_dir = os.path.join(AUDIO_DIR, curr_actor, audio_filename)
                    audio, sr = librosa.load(audio_dir, sr=SAMPLE_RATE)

                    audio, _ = librosa.effects.trim(audio, top_db=20)

                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                                n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                win_length=WIN_LENGTH, fmin=FMIN, fmax=FMAX)

                   
                    mean = np.mean(mfcc, axis=1, keepdims=True)
                    std = np.std(mfcc, axis=1, keepdims=True) + 1e-6
                    norm = (mfcc - mean) / std
                    print(norm.shape)

                    combined_data.append(norm)

                    
                    text_filename = extracted_dir.split("_")[0] + ".txt"
                    txt_label = text_filename.split("-")[2]
                    text_filename = "03" + text_filename[2:]
                    text_dir = os.path.join(TEXT_DIR, curr_actor, txt_label, text_filename)
                    text = open(text_dir, "r", encoding="utf-8").read()
                    embedding = tokenizer(
                        text,
                        max_length=64,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = embedding['input_ids'].squeeze(0)
                    attention_mask = embedding['attention_mask'].squeeze(0)
                    combined_data.append((input_ids, attention_mask))

                    torch.save(combined_data, save_path)

if __name__ == "__main__":
    preprocess_and_save(VISUAL_DIR, PROCESSED_DATA_DIR)
