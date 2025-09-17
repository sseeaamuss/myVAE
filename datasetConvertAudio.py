"""
File for converting audio dataset to spectrogram images. The data will be 
converted to 1 second clips to keep consistencey. 
"""
import os
import time
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


def process_audio_images(audio_folder: str, output_subfolder: str = "spectrograms"):
    """
    Creates a subfolder inside the given audio folder and prepares audio files
    for spectrogram processing.

    Args:
        audio_folder (str): Path to the folder containing audio files.
        output_subfolder (str): Name of the subfolder to save processed data.
    """
    # Full path to the output folder
    output_folder = os.path.join(audio_folder, output_subfolder)

    # Create the subfolder if it doesn’t exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over audio files in the folder
    for filename in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, filename)

        # Skip directories
        if not os.path.isfile(file_path):
            continue
        
        # Load audio (sr = target sample rate, e.g., 22050 Hz)
        y, sr = librosa.load("file_path", sr=22050)

        # Desired length in seconds
        target_duration = 1.0  

        # Convert to samples
        target_length = int(sr * target_duration)

        # Pad or trim to match length
        y_fixed = librosa.util.fix_length(y, size=target_length)

        print(len(y), "→", len(y_fixed))  # new length always == target_length

        print(f"Prepared {file_path} → ready for processing.")

    print(f"All files scanned. Processed data will be saved in: {output_folder}")

process_audio_images("cat-dataset/")