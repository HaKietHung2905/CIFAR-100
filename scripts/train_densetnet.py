#!/usr/bin/env python
# Simple DenseNet121 Training Script for CIFAR-100

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import pickle
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from project (adjust paths as needed)
try:
    from src.data.data_preparation import load_cifar100
    from src.models.pretrained_models import DenseNet121Model
except ImportError:
    print("Warning: Could not import from src modules. Using fallback implementations.")
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    def load_cifar100(batch_size=128, num_workers=4):
        """Fallback data loading function"""
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
    
    # Fallback DenseNet121 model
    import torchvision.models as models
    
    class DenseNet121Model(nn.Module):
        """DenseNet121 model for CIFAR-100 classification"""
        def __init__(self, num_classes=100, pretrained=True):
            super(DenseNet121Model, self).__init__()
            if pretrained:
                self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            else:
                self.model = models.densenet121(weights=None)
                
            # Replace the classifier
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
            
        def forward(self, x):
            return self.model(x)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DenseNet121 on CIFAR-100')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Train from scratch')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.set_defaults(pretrained=True)
    
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    
    return epoch_loss, epoch_acc

def main():
    """Main function to train DenseNet121 on CIFAR-100"""
    args = parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/model_analysis', exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, test_loader, classes = load_cifar100(batch_size=args.batch_size)
    
    # Split training data into training and validation
    train_size = int(0.8 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print(f"Initializing DenseNet121 (pretrained={args.pretrained})")
    model = DenseNet121Model(num_classes=100, pretrained=args.pretrained)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/densenet121_best.pth')
            print(f"Saved best model (epoch {epoch+1})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('models/densenet121_best.pth'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save results
    results = {
        'model_name': 'densenet121',
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'history': history,
        'training_time': training_time,
        'args': vars(args)
    }
   
    with open('results/model_analysis/densenet121_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to results/model_analysis/densenet121_results.pkl")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()