import librosa
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset
import torch
import numpy as np
import os

class EmbeddingDataset(Dataset):
    def __init__(self, pairs_list):
        self.pairs = pairs_list

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fpath, label = self.pairs[idx]
        # fpath now should point to a .npy file of embeddings
        embeddings = np.load(fpath)  # shape: (time_steps, feature_dim)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return embeddings_tensor, label_tensor


raw_embedding_dir = r"E:\CSC413_Data\EMBEDDING_CLIPS"
os.makedirs(raw_embedding_dir, exist_ok=True)

def trim_silence(y, top_db=20):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def precompute_embeddings(audio_clips_dir, embedding_clips_dir, model, processor):
    for actor_dir in os.listdir(audio_clips_dir):
        actor_path = os.path.join(audio_clips_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue

        actor_embedding_path = os.path.join(embedding_clips_dir, actor_dir)
        os.makedirs(actor_embedding_path, exist_ok=True)

        for emotion_dir in os.listdir(actor_path):
            emotion_path = os.path.join(actor_path, emotion_dir)
            if not os.path.isdir(emotion_path):
                continue

            emotion_embedding_path = os.path.join(actor_embedding_path, emotion_dir)
            os.makedirs(emotion_embedding_path, exist_ok=True)

            for filename in os.listdir(emotion_path):
                if filename.endswith(".wav"):
                    filepath = os.path.join(emotion_path, filename)
                    # Load audio
                    y, sr = librosa.load(filepath, sr=16000)
                    y = trim_silence(y, top_db=20)

                    # Process with Wav2Vec2 Processor
                    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
                    input_values = inputs["input_values"].to(device)
                    attention_mask = inputs.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    with torch.no_grad():
                        output = model(input_values=input_values, attention_mask=attention_mask)
                        embeddings = output.last_hidden_state.squeeze(0).cpu().numpy()
                        # embeddings shape: (time_steps, feature_dim=768)

                    # Save embeddings as .npy
                    base_name = os.path.splitext(filename)[0]  # e.g., audio_0
                    embedding_filename = base_name + ".npy"
                    embedding_filepath = os.path.join(emotion_embedding_path, embedding_filename)
                    np.save(embedding_filepath, embeddings)


if __name__ == "__main__":
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec2_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav2vec2_model.to(device)
    precompute_embeddings(r"E:\CSC413_Data\AUDIO_CLIPS", raw_embedding_dir, wav2vec2_model, processor)