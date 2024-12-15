import os
import random
import numpy as np
import librosa
import soundfile as sf
from glob import glob

raw_data_dir = r"E:\CSC413_Data\RAW_CLIPS"  # Path to your raw audio data
output_dir = r"E:\CSC413_Data\AUDIO_CLIPS"  # Directory to save extracted audio
os.makedirs(output_dir, exist_ok=True)

def trim_silence(y, top_db=20):
    # Trims leading and trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def augment_audio(raw_data_dir, output_dir) -> int:
    total_files = 0
    for actor_dir in os.listdir(raw_data_dir):
        actor_path = os.path.join(raw_data_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue

        # Prepare output path for this actor
        actor_output_path = os.path.join(output_dir, actor_dir)
        os.makedirs(actor_output_path, exist_ok=True)

        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                filepath = os.path.join(actor_path, filename)

                # Parse the filename to extract ZZ
                parts = filename.split('-')
                if len(parts) < 3:
                    continue
                ZZ = parts[2]
                ZZ_output_path = os.path.join(actor_output_path, ZZ)
                os.makedirs(ZZ_output_path, exist_ok=True)
                count = len(os.listdir(ZZ_output_path))

                # Load the audio
                y, sr = librosa.load(filepath, sr=None)
                # Trim silence from original
                y = trim_silence(y, top_db=20)

                # Save the original audio
                orig_filepath = os.path.join(ZZ_output_path, f"audio_{count}.wav")
                sf.write(orig_filepath, y, sr)
                count += 1
                total_files += 1

                # Define augmentation probabilities
                p_time_stretch = 0.5
                p_pitch_shift = 0.5
                p_noise = 0.5
                p_gain = 0.5
                p_reverb = 0.3

                # Generate a few augmented versions
                num_augmented_versions = 4

                for _ in range(num_augmented_versions):
                    y_aug = y.copy()

                    # Apply random gain (volume change)
                    if random.random() < p_gain:
                        # Gain in dB between -3dB and +3dB
                        gain_db = random.uniform(-3, 3)
                        gain_factor = 10**(gain_db/20)
                        y_aug = y_aug * gain_factor

                    # Time stretching (subtle): between 0.9 and 1.1
                    if random.random() < p_time_stretch:
                        time_stretch_factor = random.uniform(0.9, 1.1)
                        y_aug = librosa.effects.time_stretch(y_aug, rate=time_stretch_factor)

                    # Pitch shifting (subtle): between -1 and +1 semitone
                    if random.random() < p_pitch_shift:
                        pitch_shift_steps = random.uniform(-1, 1)
                        y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=pitch_shift_steps)

                    # Add noise (low level): 0.5% to 1.5%
                    if random.random() < p_noise:
                        noise_factor = random.uniform(0.005, 0.015)
                        y_aug = y_aug + noise_factor * np.random.randn(len(y_aug))

                    # Trim silence after augmentation
                    y_aug = trim_silence(y_aug, top_db=20)

                    # Ensure same length if length consistency is needed
                    # If you need consistent length across samples, you can pad/cut here.
                    # For now, we let them be variable length.

                    # Save augmented file
                    aug_filepath = os.path.join(ZZ_output_path, f"audio_{count}.wav")
                    sf.write(aug_filepath, y_aug, sr)
                    count += 1
                    total_files += 1
    return total_files

if __name__ == "__main__":
    print(augment_audio(raw_data_dir, output_dir))

