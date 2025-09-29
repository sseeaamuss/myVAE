"""
File for converting audio dataset to spectrogram images. The data will be 
converted to 1 second clips to keep consistencey. 
"""
import os
import time
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf  # to save audio files


def process_audio_images(audio_folder: str, output_subfolder: str = "spectrograms"):
    """
    Creates a subfolder inside the given audio folder, trims/pads audio files
    to 1 second, generates log spectrogram images, and saves them.

    Args:
        audio_folder (str): Path to the folder containing audio files.
        output_subfolder (str): Name of the subfolder to save spectrogram images.
    """
    # Full path to the output folder
    output_folder = os.path.join(audio_folder, output_subfolder)
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over audio files in the folder
    for filename in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, filename)

        # Skip directories
        if not os.path.isfile(file_path):
            continue
        
        try:
            # Load audio (sr = target sample rate, e.g., 22050 Hz)
            signal, sr = librosa.load(file_path, sr=22050)

            # Desired length in seconds
            target_duration = 1.0  
            target_length = int(sr * target_duration)

            # Pad or trim to match length
            signal_fixed = librosa.util.fix_length(signal, size=target_length)

            # STFT → magnitude spectrogram → log scale
            stft = librosa.stft(signal_fixed, n_fft=2048, hop_length=512)
            spectrogram = np.abs(stft)
            log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)


            # Plot spectrogram
            plt.figure(figsize=(5.12, 5.12), dpi=100)  # 5.12in × 100dpi = 512 px
            librosa.display.specshow(
                log_spectrogram,
                sr=sr,
                hop_length=512,
                x_axis='time',
                y_axis='log',
                cmap="magma"
            )
            plt.axis("off")  # hide axes for cleaner image

            # Save as PNG (same name as audio but with .png)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=100)
            plt.close()

            print(f"Processed: {file_path} → {output_path} ({len(signal)} → {len(signal_fixed)} samples)")

        except Exception as e:
            print(f"Skipping {filename}, error: {e}")

    print(f"\nAll files processed. Spectrograms saved in: {output_folder}")

process_audio_images("cat-dataset/")