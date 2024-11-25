import librosa
import numpy as np
import os

def extract_mfcc(audio_dir, sr=22050, n_mfcc=13):
    mfcc_features = []
    labels = []

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                
                # Load the audio file
                y, sr = librosa.load(file_path, sr=sr)
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                
                # Aggregate MFCCs (e.g., take the mean across time)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                mfcc_features.append(mfcc_mean)

                # Extract emotion label from the filename
                # Example filename: "01-01-01-01-01-01-05.wav"
                label = int(file.split('-')[2]) - 1  # Emotion class (0-indexed)
                labels.append(label)
    
    return np.array(mfcc_features), np.array(labels)

# Example usage
audio_dir = "/Users/edrinhasaj/Desktop/CSC413FinalProject/audio"  # Directory containing the extracted .wav files
X_audio, y_audio = extract_mfcc(audio_dir)
print(f"MFCC Features Shape: {X_audio.shape}, Labels Shape: {y_audio.shape}")

import pickle

with open("mfcc_features.pkl", "wb") as f:
    pickle.dump((X_audio, y_audio), f)
print("MFCC features and labels saved to mfcc_features.pkl")

print("Sample MFCC Feature:", X_audio[0])
print("Sample Label:", y_audio[0])
