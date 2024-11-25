import os
from moviepy.editor import VideoFileClip

def extract_audio_from_videos(video_dir, output_audio_dir):
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir)
    
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                audio_output_path = os.path.join(output_audio_dir, file.replace(".mp4", ".wav"))
                
                # Extract audio using MoviePy
                video_clip = VideoFileClip(video_path)
                video_clip.audio.write_audiofile(audio_output_path, verbose=False, logger=None)
                print(f"Extracted audio: {audio_output_path}")

# Example usage
video_dir = "/Users/edrinhasaj/Desktop/CSC413FinalProject/data"  # Path to your RAVDESS videos
output_audio_dir = "/Users/edrinhasaj/Desktop/CSC413FinalProject/audio"  # Directory to save extracted audio
extract_audio_from_videos(video_dir, output_audio_dir)

