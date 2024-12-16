import os
import random
import numpy as np
import librosa
import soundfile as sf

raw_data_dir = r"E:\CSC413_Data\RAW_CLIPS" 
output_dir = r"E:\CSC413_Data\AUDIO_CLIPS" 
os.makedirs(output_dir, exist_ok=True)

def trim_silence(y, top_db=20):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def augment_audio(raw_data_dir, output_dir) -> int:
    total_files = 0
    for actor_dir in os.listdir(raw_data_dir):
        actor_path = os.path.join(raw_data_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue

        actor_output_path = os.path.join(output_dir, actor_dir)
        os.makedirs(actor_output_path, exist_ok=True)

        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                filepath = os.path.join(actor_path, filename)

                parts = filename.split('-')
                ZZ = parts[2]
                ZZ_output_path = os.path.join(actor_output_path, ZZ)
                os.makedirs(ZZ_output_path, exist_ok=True)
                count = len(os.listdir(ZZ_output_path))

                y, sr = librosa.load(filepath, sr=None)
                y = trim_silence(y, top_db=20)

                orig_filepath = os.path.join(ZZ_output_path, f"audio_{count}.wav")
                sf.write(orig_filepath, y, sr)
                count += 1
                total_files += 1

                p_time_stretch = 0.5
                p_pitch_shift = 0.5
                p_noise = 0.5
                p_gain = 0.5

                num_augmented_versions = 4

                for _ in range(num_augmented_versions):
                    y_aug = y.copy()

                    if random.random() < p_gain:
                        gain_db = random.uniform(-3, 3)
                        gain_factor = 10**(gain_db/20)
                        y_aug = y_aug * gain_factor

                    if random.random() < p_time_stretch:
                        time_stretch_factor = random.uniform(0.9, 1.1)
                        y_aug = librosa.effects.time_stretch(y_aug, rate=time_stretch_factor)

                    if random.random() < p_pitch_shift:
                        pitch_shift_steps = random.uniform(-1, 1)
                        y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=pitch_shift_steps)

                    if random.random() < p_noise:
                        noise_factor = random.uniform(0.005, 0.015)
                        y_aug = y_aug + noise_factor * np.random.randn(len(y_aug))

                    y_aug = trim_silence(y_aug, top_db=20)

                    aug_filepath = os.path.join(ZZ_output_path, f"audio_{count}.wav")
                    sf.write(aug_filepath, y_aug, sr)
                    count += 1
                    total_files += 1
    return total_files

if __name__ == "__main__":
    print(augment_audio(raw_data_dir, output_dir))

