import os
from PIL import Image

def resize_to_square(folder, output_folder=None, size=None):
    """
    Resizes all PNGs in a folder to be square by taking the smaller side.

    Args:
        folder (str): Input folder with PNGs.
        output_folder (str, optional): Folder to save processed images.
                                       If None, overwrites in place.
        size (int, optional): Final square size. If None, use min(width, height).
    """
    if output_folder is None:
        output_folder = folder
    else:
        os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(folder, filename)
            img = Image.open(filepath)
            width, height = img.size

            # Target size (square)
            target_size = size if size else min(width, height)

            # Resize to square
            img_resized = img.resize((target_size, target_size), Image.LANCZOS)

            save_path = os.path.join(output_folder, filename)
            img_resized.save(save_path)

            print(f"{filename}: {width}x{height} → {target_size}x{target_size}")

resize_to_square('test-spectrograms', size = 256)