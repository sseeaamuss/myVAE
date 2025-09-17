import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import io

from PIL import Image
from tqdm import tqdm


from matplotlib import pyplot as plt

import requests
import tarfile
import os
import time

import pyarrow as pa
import lance


vae_config = {
    "BATCH_SIZE": 4096,
    "IN_RESOLUTION": 32,
    "IN_CHANNELS": 3,
    "NUM_EPOCHS": 100,
    "LEARNING_RATE": 1e-4,
    "HIDDEN_DIMS": [64, 128, 256, 512],
    "LATENT_DIM_SIZE": 64,
}

# Define the URL for the dataset file
data_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"

# Create the data directory if it doesn't exist
data_dir = "cinic-10-data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download the dataset file
print("Downloading CINIC-10 dataset...")
data_file = os.path.join(data_dir, "CINIC-10.tar.gz")

response = requests.get(data_url, stream=True)
total_size = int(response.headers.get('content-length', 0))
block_size = 1024

start_time = time.time()
progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

with open(data_file, 'wb') as f:
    for chunk in response.iter_content(chunk_size=block_size):
        if chunk:
            f.write(chunk)
            progress_bar.update(len(chunk))

end_time = time.time()
download_time = end_time - start_time
progress_bar.close()

print(f"\nDownload time: {download_time:.2f} seconds")

# Extract the dataset files
print("Extracting dataset files...")
with tarfile.open(data_file, 'r:gz') as tar:
    tar.extractall(path=data_dir)

print("Dataset downloaded and extracted successfully!")