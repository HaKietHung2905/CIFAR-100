import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def load_cifar100(batch_size=128, num_workers=4):
    """
    Load and prepare the CIFAR-100 dataset with appropriate transforms
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        
    Returns:
        train_loader, test_loader, classes
    """
    # Define transforms for training data (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Define transforms for testing data (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load CIFAR-100 dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', 
        train=True,
        download=True, 
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', 
        train=False,
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Get class names
    classes = train_dataset.classes
    
    return train_loader, test_loader, classes

def visualize_data_samples(loader, classes, num_images=10):
    """
    Visualize some sample images from the dataset
    
    Args:
        loader: DataLoader for the dataset
        classes: List of class names
        num_images: Number of images to display
    """
    # Get a batch of training data
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Convert images for display
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Unnormalize images
    images_np = images.numpy()
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    
    # Transpose from [B, C, H, W] to [B, H, W, C]
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    # Unnormalize
    images_np = std * images_np + mean
    images_np = np.clip(images_np, 0, 1)
    
    # Plot the images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_images):
        axes[i].imshow(images_np[i])
        axes[i].set_title(f"Class: {classes[labels[i]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar100_samples.png')
    plt.show()

if __name__ == "__main__":
    # Example usage
    train_loader, test_loader, classes = load_cifar100()
    print(f"Dataset loaded: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples")
    print(f"Number of classes: {len(classes)}")
    
    # Visualize some samples
    visualize_data_samples(train_loader, classes)