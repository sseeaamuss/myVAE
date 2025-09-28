import os
from PIL import Image

# Path to your folder
folder = "cat-spectrograms"

# Loop through all files in the folder
for filename in os.listdir(folder):
    if filename.lower().endswith(".png"):  # only PNGs
        filepath = os.path.join(folder, filename)
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                print(f"{filename}: {width}x{height}")
        except Exception as e:
            print(f"Could not open {filename}: {e}")