import os
import random

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf

def extract_text(model, raw_data_dir, output_dir) -> int:
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
                if len(parts) < 3:
                    continue
                ZZ = parts[2]
                ZZ_output_path = os.path.join(actor_output_path, ZZ)
                os.makedirs(ZZ_output_path, exist_ok=True)
                count = len(os.listdir(ZZ_output_path))

                y, sr = librosa.load(filepath, sr=None)
                data = librosa.to_mono(y)
                result = pipe(data)

                out_filepath = os.path.join(ZZ_output_path, f"{filename.replace(".wav", ".txt")}")
                f = open(out_filepath, "x", encoding="utf-8")
                f.write(result["text"])
                f.close()
                total_files += 1
    return total_files

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    audio_dir = r"E:\CSC413_Data\RAW_CLIPS"
    output_dir = r"E:\CSC413_Data\EXTRACTED_TEXT"

    extract_text(pipe, audio_dir, output_dir)
