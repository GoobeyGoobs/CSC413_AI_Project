import os
import librosa
import numpy as np

def extract_mfcc_features(input_dir, output_dir):
    ORIGINAL_ROOT = input_dir 
    TRANSFORMED_ROOT = output_dir 

    SAMPLE_RATE = 22050
    N_MFCC = 48
    N_FFT = 2048
    HOP_LENGTH = 512
    WIN_LENGTH = 2048
    FMIN = 50
    FMAX = 8000

    if not os.path.exists(TRANSFORMED_ROOT):
        os.makedirs(TRANSFORMED_ROOT)

    for actor_dir in os.listdir(ORIGINAL_ROOT):
        actor_path = os.path.join(ORIGINAL_ROOT, actor_dir)
        if os.path.isdir(actor_path):
            transformed_actor_path = os.path.join(TRANSFORMED_ROOT, actor_dir)
            if not os.path.exists(transformed_actor_path):
                os.makedirs(transformed_actor_path)

            for label_dir in os.listdir(actor_path):
                label_path = os.path.join(actor_path, label_dir)
                if os.path.isdir(label_path):
                    transformed_label_path = os.path.join(transformed_actor_path, label_dir)
                    if not os.path.exists(transformed_label_path):
                        os.makedirs(transformed_label_path)

                    for wav_file in os.listdir(label_path):
                        if wav_file.lower().endswith('.wav'):
                            wav_path = os.path.join(label_path, wav_file)
                            audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

                            audio, _ = librosa.effects.trim(audio, top_db=20)

                            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                                        n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                        win_length=WIN_LENGTH, fmin=FMIN, fmax=FMAX)

                            mean = np.mean(mfcc, axis=1, keepdims=True)
                            std = np.std(mfcc, axis=1, keepdims=True) + 1e-6
                            norm = (mfcc - mean) / std

                            mfcc_save_path = os.path.join(transformed_label_path, wav_file.replace('.wav', '.npy'))
                            np.save(mfcc_save_path, norm)

if __name__ == "__main__":
    audio_dir = r"E:\CSC413_Data\AUDIO_CLIPS"  
    output_dir = r"E:\CSC413_Data\ORDERED_DATA"  
    extract_mfcc_features(audio_dir, output_dir)
