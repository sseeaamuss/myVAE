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



print(dir(lance))
print(lance.__version__)

def process_images(images_folder, split, schema):

    # Iterate over the categories within each data type
    label_folder = os.path.join(images_folder, split)
    for label in os.listdir(label_folder):
        label_folder = os.path.join(images_folder, split, label)
        
        # Iterate over the images within each label
        for filename in tqdm(os.listdir(label_folder), desc=f"Processing {split} - {label}"):
            # Construct the full path to the image
            image_path = os.path.join(label_folder, filename)

            # Read and convert the image to a binary format
            with open(image_path, 'rb') as f:
                binary_data = f.read()

            image_array = pa.array([binary_data], type=pa.binary())
            filename_array = pa.array([filename], type=pa.string())
            label_array = pa.array([label], type=pa.string())
            split_array = pa.array([split], type=pa.string())

            # Yield RecordBatch for each image
            yield pa.RecordBatch.from_arrays(
                [image_array, filename_array, label_array, split_array],
                schema=schema
            )

# Function to write PyArrow Table to Lance dataset
def write_to_lance(images_folder, dataset_name, schema):
    for split in ['test', 'train', 'valid']:
        lance_file_path = os.path.join(images_folder, f"{dataset_name}_{split}.lance")
        
        reader = pa.RecordBatchReader.from_batches(schema, process_images(images_folder, split, schema))
        lance.write_dataset(
            reader,
            lance_file_path,
            schema,
        )



dataset_path = "cat-spectrograms"
dataset_name = os.path.basename(dataset_path)

start = time.time()
schema = pa.schema([
    pa.field("image", pa.binary()),
    pa.field("filename", pa.string()),
    pa.field("label", pa.string()),
    pa.field("split", pa.string())
])

start = time.time()
write_to_lance(dataset_path, dataset_name, schema)
end = time.time()
print(f"Time(sec): {end - start:.2f}")