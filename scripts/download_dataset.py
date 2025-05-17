#!/usr/bin/env python
# CIFAR-100 Image Classification Project - Dataset Download Script

import os
import sys
import torchvision
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import from the src directory
from src.data.data_preparation import load_cifar100

def download_cifar100(download_path="./data", force_download=False):
    """
    Download the CIFAR-100 dataset to the specified directory.
    
    Args:
        download_path: Directory where the dataset will be saved
        force_download: Whether to force download even if the dataset already exists
    """
    print(f"Downloading CIFAR-100 dataset to {download_path}...")
    
    # Create the directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Download the training set
    train_dataset = torchvision.datasets.CIFAR100(
        root=download_path,
        train=True,
        download=True if force_download else not os.path.exists(os.path.join(download_path, 'cifar-100-python'))
    )
    
    # Download the test set
    test_dataset = torchvision.datasets.CIFAR100(
        root=download_path,
        train=False,
        download=True if force_download else not os.path.exists(os.path.join(download_path, 'cifar-100-python'))
    )
    
    print(f"CIFAR-100 dataset downloaded successfully.")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Print dataset information
    classes = train_dataset.classes
    print(f"Number of classes: {len(classes)}")
    
    # Print a few class names as example
    print("Sample classes:")
    for i in range(min(10, len(classes))):
        print(f"  - {classes[i]}")
    
    # Print dataset structure
    print("\nDataset structure:")
    print(f"{download_path}/")
    print(f"└── cifar-100-python/")
    print(f"    ├── meta")
    print(f"    ├── test")
    print(f"    └── train")
    
    return train_dataset, test_dataset


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download CIFAR-100 dataset')
    parser.add_argument('--path', type=str, default='./data',
                        help='Directory to save the dataset (default: ./data)')
    parser.add_argument('--force', action='store_true',
                        help='Force download even if dataset already exists')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_cifar100(args.path, args.force)