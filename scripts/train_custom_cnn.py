#!/usr/bin/env python
# CIFAR-100 Image Classification Project - Custom CNN Training Script

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import yaml

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project modules
from src.models.custom_cnn import CustomCNN
from src.data.data_preparation import load_cifar100


def load_config(config_file):
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to the YAML configuration file
        
    Returns:
        config: Dictionary with configuration parameters
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found. Using default settings.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return {}


def create_data_loaders(batch_size=128, train_val_split=0.8, seed=42, augmentation=True):
    """
    Create data loaders for CIFAR-100
    
    Args:
        batch_size: Batch size for training and testing
        train_val_split: Fraction of training data to use for training (rest for validation)
        seed: Random seed for reproducibility
        augmentation: Whether to use data augmentation
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        classes: List of class names
    """
    # Define transforms for training data (with augmentation)
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    # Define transforms for validation and test data (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load CIFAR-100 datasets
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
        transform=val_transform
    )
    
    # Split training data into training and validation
    train_size = int(train_val_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # For validation, we need to override the transform
    val_dataset = torchvision.datasets.CIFAR100(
        root='./data', 
        train=True,
        download=False, 
        transform=val_transform
    )
    _, val_dataset = random_split(
        val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Get class names
    classes = test_dataset.classes
    
    return train_loader, val_loader, test_loader, classes


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        
    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / total_samples, 
            'acc': running_corrects / total_samples
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda/cpu)
        
    Returns:
        epoch_loss: Average loss for the validation set
        epoch_acc: Accuracy for the validation set
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Disable gradients
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total_samples, 
                'acc': running_corrects / total_samples
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, config, device):
    """
    Train the model with the specified configuration
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Dictionary with training configuration
        device: Device to train on (cuda/cpu)
        
    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    # Set model to device
    model = model.to(device)
    
    # Extract parameters from config (convert to appropriate types)
    num_epochs = int(config.get('num_epochs', 50))
    learning_rate = float(config.get('learning_rate', 0.001))
    weight_decay = float(config.get('weight_decay', 1e-4))
    patience = int(config.get('patience', 7))
    optimizer_name = str(config.get('optimizer', 'adam')).lower()
    scheduler_name = str(config.get('scheduler', 'reduce_on_plateau')).lower()
    model_name = str(config.get('model_name', 'custom_cnn'))
    checkpoint_interval = int(config.get('checkpoint_interval', 5))
    
    # Print parameters for debugging
    print(f"Training parameters:")
    print(f"  num_epochs: {num_epochs} ({type(num_epochs)})")
    print(f"  learning_rate: {learning_rate} ({type(learning_rate)})")
    print(f"  weight_decay: {weight_decay} ({type(weight_decay)})")
    print(f"  optimizer: {optimizer_name} ({type(optimizer_name)})")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = float(config.get('momentum', 0.9))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                             momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    # Define learning rate scheduler
    if scheduler_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=0
        )
    elif scheduler_name == 'step':
        step_size = int(config.get('step_size', 10))
        gamma = float(config.get('gamma', 0.1))
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_name == 'none' or scheduler_name == 'null':
        scheduler = None
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported")
    
    # Initialize training variables
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    # Create directories for checkpoints and results
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'time_per_epoch': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - Learning Rate: {current_lr:.6f}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler is not None:
            if scheduler_name == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        history['time_per_epoch'].append(epoch_time)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint if needed
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), f"models/{model_name}_epoch{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Check if this is the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f"models/{model_name}_best.pth"))
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history


def evaluate_model(model, test_loader, classes, config, device):
    """
    Evaluate the model on the test set
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        classes: List of class names
        config: Dictionary with configuration parameters
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        results: Dictionary with evaluation results
    """
    model = model.to(device)
    model.eval()
    model_name = str(config.get('model_name', 'custom_cnn'))
    
    # Initialize variables
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    total_samples = 0
    
    # Disable gradients
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            total_samples += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': running_loss / total_samples})
    
    # Calculate metrics
    test_loss = running_loss / total_samples
    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    all_probs = np.concatenate(all_probs)
    
    # Calculate top-5 accuracy
    top5_indices = np.argsort(-all_probs, axis=1)[:, :5]
    top5_correct = 0
    for i, label in enumerate(all_labels):
        if label in top5_indices[i]:
            top5_correct += 1
    top5_accuracy = top5_correct / len(all_labels)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    # Calculate per-class precision, recall, and F1 score
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Store results
    results = {
        'model_name': model_name,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support,
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }
    
    # Print results
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy (Top-1): {test_accuracy:.4f}")
    print(f"Test Accuracy (Top-5): {top5_accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    
    # Save detailed per-class metrics
    df_metrics = pd.DataFrame({
        'Class': classes,
        'Precision': per_class_precision,
        'Recall': per_class_recall,
        'F1 Score': per_class_f1,
        'Support': per_class_support
    })
    
    # Save to CSV
    os.makedirs('results/model_analysis', exist_ok=True)
    df_metrics.to_csv(f"results/model_analysis/{model_name}_class_metrics.csv", index=False)
    
    # Save results
    with open(f"results/model_analysis/{model_name}_results.pkl", 'wb') as f:
        import pickle
        pickle.dump(results, f)
    
    return results


def plot_training_history(history, config):
    """
    Plot training history
    
    Args:
        history: Dictionary with training history
        config: Dictionary with configuration parameters
    """
    model_name = str(config.get('model_name', 'custom_cnn'))
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"results/plots/{model_name}_training_history.png", dpi=200)
    plt.close()
    
    # Plot learning rate
    plt.figure(figsize=(10, 5))
    plt.semilogy(history['learning_rates'])
    plt.title(f'{model_name} - Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/plots/{model_name}_learning_rate.png", dpi=200)
    plt.close()
    
    # Save history to CSV
    epochs = list(range(1, len(history['train_loss']) + 1))
    df_history = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': history['train_loss'],
        'Val Loss': history['val_loss'],
        'Train Acc': history['train_acc'],
        'Val Acc': history['val_acc'],
        'Learning Rate': history['learning_rates'],
        'Time (s)': history['time_per_epoch']
    })
    
    df_history.to_csv(f"results/plots/{model_name}_training_history.csv", index=False)


def create_config_file():
    """Create a default configuration file for Custom CNN"""
    config = {
        'model_name': 'custom_cnn',
        'num_epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'optimizer': 'adam',
        'scheduler': 'reduce_on_plateau',
        'patience': 7,
        'checkpoint_interval': 5,
        'augmentation': {
            'enabled': True,
            'random_crop': True,
            'random_horizontal_flip': True,
            'color_jitter': True
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs('configs', exist_ok=True)
    
    # Save to YAML file
    with open('configs/custom_cnn_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Created default configuration file: configs/custom_cnn_config.yaml")
    
    return config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train and evaluate Custom CNN on CIFAR-100')
    parser.add_argument('--config', type=str, default='configs/custom_cnn_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training and testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on the test set')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the model checkpoint for evaluation')
    parser.add_argument('--create-config', action='store_true',
                        help='Create a default configuration file')
    
    args = parser.parse_args()
    
    # Create default config file if requested
    if args.create_config:
        config = create_config_file()
        return
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load configuration
    try:
        config = load_config(args.config)
    except:
        print(f"Failed to load configuration from {args.config}. Creating default configuration.")
        config = create_config_file()
    
    # Update config with command-line arguments
    config['batch_size'] = args.batch_size
    config['model_name'] = 'custom_cnn'
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value} ({type(value)})")
    
    # Create data loaders
    if 'augmentation' in config:
        augmentation = bool(config['augmentation'].get('enabled', True))
    else:
        augmentation = True
    
    train_loader, val_loader, test_loader, classes = create_data_loaders(
        batch_size=int(config['batch_size']),
        train_val_split=0.8,
        seed=args.seed,
        augmentation=augmentation
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_loader.dataset)}")
    print(f"  Validation: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    print(f"  Number of classes: {len(classes)}")
    
    # Create model
    model = CustomCNN(num_classes=100)
    
    print(f"\nCreated {config['model_name']} model")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train or load model
    if args.eval_only:
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = f"models/{config['model_name']}_best.pth"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"\nLoaded model from {model_path}")
        else:
            print(f"Error: Model checkpoint {model_path} not found")
            return
    else:
        # Train the model
        print("\nTraining model...")
        model, history = train_model(model, train_loader, val_loader, config, device)
        
        # Plot training history
        plot_training_history(history, config)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, test_loader, classes, config, device)
    
    print("\nDone!")


if __name__ == "__main__":
    main()