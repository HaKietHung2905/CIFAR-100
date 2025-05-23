import argparse
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sys
from tqdm import tqdm
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Fixed imports - only import what actually exists
from src.models.efficientnet_models import (
    get_efficientnet_model, 
    EFFICIENTNET_CONFIGS, 
    get_optimal_config
)

try:
    from src.data.data_preparation import load_cifar100
    DATA_LOADER_AVAILABLE = True
except ImportError:
    print("Warning: data_preparation not available, will use fallback")
    DATA_LOADER_AVAILABLE = False

from sklearn.metrics import classification_report, confusion_matrix

# Fallback data loading function
def fallback_load_cifar100(batch_size=128):
    """Fallback function to load CIFAR-100"""
    import torchvision
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = trainset.classes
    return train_loader, test_loader, classes


class EfficientNetTrainer:
    """Fixed trainer for EfficientNet models on CIFAR-100"""
    
    def __init__(self, variant='b0', config=None, device=None):
        self.variant = variant
        self.config = config or EFFICIENTNET_CONFIGS[variant]
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Add target_size to config if not present (for backward compatibility)
        if 'target_size' not in self.config:
            self.config['target_size'] = 224
        
        # Initialize tracking variables
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Create directories
        os.makedirs('results/efficientnet', exist_ok=True)
        os.makedirs('models/efficientnet', exist_ok=True)
        
    def create_data_loaders(self, train_ratio=0.8):
        
        # Enhanced data augmentation for EfficientNet - FIXED ORDER
        train_transform = transforms.Compose([
            transforms.Resize((self.config['target_size'], self.config['target_size'])),
            transforms.RandomCrop(self.config['target_size'], padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),  # ← MOVED BEFORE RandomErasing
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # ← MOVED AFTER ToTensor
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Resize((self.config['target_size'], self.config['target_size'])),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Load datasets
        if DATA_LOADER_AVAILABLE:
            train_loader, test_loader, classes = load_cifar100(
                batch_size=self.config['batch_size']
            )
        else:
            train_loader, test_loader, classes = fallback_load_cifar100(
                batch_size=self.config['batch_size']
            )
        
        # Override transforms
        train_dataset = train_loader.dataset
        train_dataset.transform = train_transform
        
        test_dataset = test_loader.dataset
        test_dataset.transform = val_test_transform
        
        # Create train/validation split
        train_size = int(train_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders - REDUCED num_workers to avoid issues
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # ← CHANGED TO 0 to avoid multiprocessing issues
            pin_memory=False,  # ← CHANGED TO False
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # ← CHANGED TO 0
            pin_memory=False  # ← CHANGED TO False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # ← CHANGED TO 0
            pin_memory=False  # ← CHANGED TO False
        )
        
        return train_loader, val_loader, test_loader, classes
    
    def create_model(self, pretrained=True, freeze_backbone=False):
        """Create EfficientNet model - fixed to use only available parameters"""
        model = get_efficientnet_model(
            variant=self.variant,
            num_classes=100,
            pretrained=pretrained,
            dropout_rate=self.config['dropout_rate']
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in model.model.features.parameters():
                param.requires_grad = False
        
        return model.to(self.device)
    
    def create_optimizer_scheduler(self, model):
        """Create optimizer and learning rate scheduler"""
        
        # Simple optimizer setup (removed complex parameter grouping)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        return optimizer, scheduler
    
    def warmup_learning_rate(self, optimizer, epoch, warmup_epochs):
        """Apply learning rate warmup"""
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == targets.data).item()
            total_samples += targets.size(0)
            
            # Update progress bar
            current_loss = running_loss / total_samples
            current_acc = running_corrects / total_samples
            
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion, epoch):
        """Validate for one epoch"""
        model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == targets.data).item()
                total_samples += targets.size(0)
                
                # Update progress bar
                current_loss = running_loss / total_samples
                current_acc = running_corrects / total_samples
                
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        return epoch_loss, epoch_acc
    
    def train_model(self, pretrained=True, freeze_backbone=False, save_best=True):
        """Complete training loop"""
        
        print(f"Training EfficientNet-{self.variant.upper()} on CIFAR-100")
        print(f"Device: {self.device}")
        print(f"Configuration: {self.config}")
        print("-" * 60)
        
        # Create data loaders
        train_loader, val_loader, test_loader, classes = self.create_data_loaders()
        
        print(f"Dataset sizes:")
        print(f"  Train: {len(train_loader.dataset)}")
        print(f"  Validation: {len(val_loader.dataset)}")
        print(f"  Test: {len(test_loader.dataset)}")
        print("-" * 60)
        
        # Create model
        model = self.create_model(pretrained=pretrained, freeze_backbone=freeze_backbone)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print("-" * 60)
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_scheduler(model)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training variables
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Warmup learning rate
            if epoch < self.config['warmup_epochs']:
                self.warmup_learning_rate(optimizer, epoch, self.config['warmup_epochs'])
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion, epoch)
            
            # Update scheduler
            if epoch >= self.config['warmup_epochs']:
                scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config['epochs']} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                if save_best:
                    model_path = f"models/efficientnet/efficientnet_{self.variant}_best.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_acc': best_val_acc,
                        'config': self.config
                    }, model_path)
                    print(f"  Saved best model (Val Acc: {best_val_acc:.4f})")
            else:
                patience_counter += 1
            
            print("-" * 60)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Unfreeze backbone after warmup (if it was frozen)
            if epoch == self.config['warmup_epochs'] and freeze_backbone:
                for param in model.model.features.parameters():
                    param.requires_grad = True
                print("Unfroze backbone for fine-tuning")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Load best model for testing
        if save_best:
            checkpoint = torch.load(f"models/efficientnet/efficientnet_{self.variant}_best.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, test_loader, classes
    
    def evaluate_model(self, model, test_loader, classes):
        """Comprehensive model evaluation"""
        model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        print("Evaluating model on test set...")
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Testing"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        test_acc = np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Top-5 accuracy
        all_probs = np.array(all_probs)
        top5_preds = np.argsort(-all_probs, axis=1)[:, :5]
        top5_acc = np.mean([all_targets[i] in top5_preds[i] for i in range(len(all_targets))])
        
        # Classification report
        class_report = classification_report(
            all_targets, all_preds, 
            target_names=classes, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        print(f"\nTest Results:")
        print(f"  Top-1 Accuracy: {test_acc:.4f}")
        print(f"  Top-5 Accuracy: {top5_acc:.4f}")
        print(f"  Macro F1-Score: {class_report['macro avg']['f1-score']:.4f}")
        print(f"  Weighted F1-Score: {class_report['weighted avg']['f1-score']:.4f}")
        
        return {
            'test_acc': test_acc,
            'top5_acc': top5_acc,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'classification_report': class_report,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title(f'EfficientNet-{self.variant.upper()} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Train')
        plt.plot(self.val_accuracies, label='Validation')
        plt.title(f'EfficientNet-{self.variant.upper()} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 3, 3)
        plt.plot(self.learning_rates)
        plt.title(f'EfficientNet-{self.variant.upper()} - Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/efficientnet/training_history_{self.variant}.png', dpi=300, bbox_inches='tight')
        plt.show()


def compare_efficientnet_variants(variants=['b0', 'b1', 'b2'], device=None):
    """Compare multiple EfficientNet variants"""
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_summary = []
    
    for variant in variants:
        print(f"\n{'='*80}")
        print(f"Training EfficientNet-{variant.upper()}")
        print(f"{'='*80}")
        
        try:
            # Get optimal configuration based on available memory
            config = get_optimal_config(variant, 8)  # Assume 8GB
            
            # Create trainer
            trainer = EfficientNetTrainer(variant=variant, config=config, device=device)
            
            # Train model
            model, test_loader, classes = trainer.train_model(
                pretrained=True, 
                freeze_backbone=(variant in ['b2', 'b3'])  # Freeze for larger models
            )
            
            # Evaluate model
            results = trainer.evaluate_model(model, test_loader, classes)
            
            # Plot results
            trainer.plot_training_history()
            
            # Add to summary
            results_summary.append({
                'Variant': variant.upper(),
                'Top-1 Acc': f"{results['test_acc']:.4f}",
                'Top-5 Acc': f"{results['top5_acc']:.4f}",
                'Macro F1': f"{results['classification_report']['macro avg']['f1-score']:.4f}",
                'Parameters': f"{sum(p.numel() for p in model.parameters()):,}",
                'Batch Size': config['batch_size']
            })
            
            # Clean up memory
            del model, trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error training {variant}: {str(e)}")
            continue
    
    # Create comparison table
    if results_summary:
        comparison_df = pd.DataFrame(results_summary)
        
        print(f"\n{'='*80}")
        print("EFFICIENTNET VARIANTS COMPARISON")
        print(f"{'='*80}")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('results/efficientnet/comparison_summary.csv', index=False)
    
    return results_summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train EfficientNet on CIFAR-100')
    parser.add_argument('--variant', type=str, default='b0', 
                       choices=['b0', 'b1', 'b2', 'b3'],
                       help='EfficientNet variant to train')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple variants')
    parser.add_argument('--variants', nargs='+', default=['b0', 'b1', 'b2'],
                       help='Variants to compare when using --compare')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Train from scratch without pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone during training')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.compare:
        # Compare multiple variants
        compare_efficientnet_variants(args.variants)
    else:
        # Train single variant
        config = EFFICIENTNET_CONFIGS[args.variant].copy()
        
        # Override config with command line arguments
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.lr:
            config['learning_rate'] = args.lr
        
        # Create trainer
        trainer = EfficientNetTrainer(variant=args.variant, config=config)
        
        # Train model
        model, test_loader, classes = trainer.train_model(
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone
        )
        
        # Evaluate model
        results = trainer.evaluate_model(model, test_loader, classes)
        
        # Plot and save results
        trainer.plot_training_history()


if __name__ == "__main__":
    main()