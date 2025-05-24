#!/usr/bin/env python
# Simple ConvNeXt Training Script for CIFAR-100
# Following the DenseNet training pattern

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
    from src.models.pretrained_models import get_convnext_model
except ImportError:
    print("Warning: Could not import from src modules. Using fallback implementations.")
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision.models as models
    
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
    
    # Fallback ConvNeXt model implementation
    class ConvNeXtModel(nn.Module):
        """ConvNeXt model for CIFAR-100 classification"""
        def __init__(self, variant='tiny', num_classes=100, pretrained=True):
            super(ConvNeXtModel, self).__init__()
            
            if variant == 'tiny':
                if pretrained:
                    self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
                else:
                    self.model = models.convnext_tiny(weights=None)
                
                # Modify for CIFAR-100 (32x32 -> 8x8)
                self.model.features[0][0] = nn.Conv2d(3, 96, kernel_size=4, stride=4)
                # Replace classifier
                self.model.classifier[2] = nn.Linear(768, num_classes)
                
            elif variant == 'small':
                if pretrained:
                    self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
                else:
                    self.model = models.convnext_small(weights=None)
                
                # Modify for CIFAR-100
                self.model.features[0][0] = nn.Conv2d(3, 96, kernel_size=4, stride=4)
                self.model.classifier[2] = nn.Linear(768, num_classes)
                
            elif variant == 'base':
                if pretrained:
                    self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
                else:
                    self.model = models.convnext_base(weights=None)
                
                # Modify for CIFAR-100
                self.model.features[0][0] = nn.Conv2d(3, 128, kernel_size=4, stride=4)
                self.model.classifier[2] = nn.Linear(1024, num_classes)
            else:
                raise ValueError(f"Unsupported variant: {variant}")
                
        def forward(self, x):
            return self.model(x)
    
    def get_convnext_model(variant='tiny', num_classes=100, pretrained=True, custom_cifar=False):
        """Fallback model creation function"""
        return ConvNeXtModel(variant=variant, num_classes=num_classes, pretrained=pretrained)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ConvNeXt on CIFAR-100')
    parser.add_argument('--variant', type=str, default='tiny', 
                       choices=['tiny', 'small', 'base'],
                       help='ConvNeXt variant to use (default: tiny)')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=4e-3, 
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, 
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, 
                       help='Early stopping patience')
    parser.add_argument('--custom-cifar', action='store_true',
                       help='Use custom CIFAR-adapted ConvNeXt')
    parser.add_argument('--pretrained', action='store_true', 
                       help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', 
                       help='Train from scratch')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                       help='Number of warmup epochs')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--mixup-alpha', type=float, default=0.8,
                       help='Mixup alpha parameter')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all ConvNeXt variants')
    parser.set_defaults(pretrained=True)
    
    return parser.parse_args()


def apply_mixup(inputs, targets, alpha=0.8):
    """Apply mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)
    
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    targets_a, targets_b = targets, targets[index]
    
    return mixed_inputs, targets_a, targets_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cosine_lr_schedule(optimizer, epoch, total_epochs, base_lr, min_lr=1e-6, warmup_epochs=10):
    """Cosine learning rate schedule with warmup"""
    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * epoch / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def train_epoch(model, dataloader, criterion, optimizer, device, args, epoch):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    # Apply learning rate schedule
    current_lr = cosine_lr_schedule(optimizer, epoch, args.epochs, args.lr, 
                                   warmup_epochs=args.warmup_epochs)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply mixup after warmup period
        use_mixup = args.mixup_alpha > 0 and epoch >= args.warmup_epochs
        
        if use_mixup and np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = apply_mixup(inputs, labels, args.mixup_alpha)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{running_loss/total:.4f}",
            'Acc': f"{running_corrects/total:.4f}",
            'LR': f"{current_lr:.6f}"
        })
    
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    
    return epoch_loss, epoch_acc, current_lr


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


def calculate_top5_accuracy(model, dataloader, device):
    """Calculate top-5 accuracy"""
    model.eval()
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Calculating Top-5 Accuracy"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get top-5 predictions
            _, top5_preds = torch.topk(outputs, k=5, dim=1)
            
            # Check if true label is in top-5 predictions
            for i in range(len(labels)):
                if labels[i] in top5_preds[i]:
                    top5_correct += 1
            total += labels.size(0)
    
    top5_accuracy = top5_correct / total
    return top5_accuracy


def train_single_model(args):
    """Train a single ConvNeXt model"""
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
    
    # Adjust batch size based on variant
    variant_batch_sizes = {'tiny': 128, 'small': 64, 'base': 32}
    if args.batch_size == 64:  # Using default
        args.batch_size = variant_batch_sizes.get(args.variant, 64)
    
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
    impl_name = "custom" if args.custom_cifar else "pretrained"
    print(f"Initializing ConvNeXt-{args.variant} ({impl_name}, pretrained={args.pretrained})")
    
    model = get_convnext_model(
        variant=args.variant,
        num_classes=100,
        pretrained=args.pretrained,
        custom_cifar=args.custom_cifar
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    print(f"Configuration:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Mixup alpha: {args.mixup_alpha}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []
    }
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc, current_lr = train_epoch(
            model, train_loader, criterion, optimizer, device, args, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_name = f"convnext_{args.variant}_{impl_name}_best.pth"
            torch.save(model.state_dict(), f'models/{model_name}')
            print(f"✅ Saved best model (epoch {epoch+1}) with val_acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience and epoch >= args.warmup_epochs:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    model_name = f"convnext_{args.variant}_{impl_name}_best.pth"
    model.load_state_dict(torch.load(f'models/{model_name}'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    # Calculate top-5 accuracy
    top5_acc = calculate_top5_accuracy(model, test_loader, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy (Top-1): {test_acc:.4f}")
    print(f"Test Accuracy (Top-5): {top5_acc:.4f}")
    
    # Save results
    results = {
        'model_name': f'convnext_{args.variant}_{impl_name}',
        'variant': args.variant,
        'implementation': impl_name,
        'test_accuracy': test_acc,
        'top5_accuracy': top5_acc,
        'test_loss': test_loss,
        'best_val_accuracy': best_val_acc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'history': history,
        'training_time': training_time,
        'args': vars(args)
    }
    
    result_file = f'results/model_analysis/convnext_{args.variant}_{impl_name}_results.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {result_file}")
    
    return results


def compare_all_variants(args):
    """Compare all ConvNeXt variants"""
    print("=" * 80)
    print("COMPARING ALL CONVNEXT VARIANTS")
    print("=" * 80)
    
    variants = ['tiny', 'small', 'base']
    implementations = [False, True]  # Pretrained vs Custom
    results_summary = []
    
    # Shorter training for comparison
    original_epochs = args.epochs
    args.epochs = min(50, args.epochs)  # Limit to 50 epochs for comparison
    
    for variant in variants:
        for custom_cifar in implementations:
            impl_name = "custom" if custom_cifar else "pretrained"
            print(f"\n{'-'*60}")
            print(f"Training ConvNeXt-{variant} ({impl_name})")
            print(f"{'-'*60}")
            
            # Update args for current variant
            args.variant = variant
            args.custom_cifar = custom_cifar
            
            # Adjust batch size based on variant
            variant_batch_sizes = {'tiny': 128, 'small': 64, 'base': 32}
            args.batch_size = variant_batch_sizes[variant]
            
            try:
                results = train_single_model(args)
                results_summary.append(results)
                
                # Clean up memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ Error training {variant} ({impl_name}): {e}")
                continue
    
    # Restore original epochs
    args.epochs = original_epochs
    
    # Create comparison report
    if results_summary:
        create_comparison_report(results_summary)
    
    return results_summary


def create_comparison_report(results_summary):
    """Create a comprehensive comparison report"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create DataFrame
    df_data = []
    for result in results_summary:
        df_data.append({
            'Model': f"{result['variant']}-{result['implementation']}",
            'Variant': result['variant'],
            'Implementation': result['implementation'],
            'Test Accuracy': result['test_accuracy'],
            'Top-5 Accuracy': result['top5_accuracy'],
            'Parameters (M)': result['total_params'] / 1e6,
            'Training Time (min)': result['training_time'] / 60
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('Test Accuracy', ascending=False)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/convnext_comparison_results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("CONVNEXT COMPARISON RESULTS")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Find best models
    best_accuracy = df.loc[df['Test Accuracy'].idxmax()]
    best_efficiency = df.loc[(df['Test Accuracy'] / df['Parameters (M)']).idxmax()]
    
    print(f"\n🏆 Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Test Accuracy']:.4f})")
    print(f"⚡ Most Efficient: {best_efficiency['Model']} "
          f"({best_efficiency['Test Accuracy']:.4f} acc, {best_efficiency['Parameters (M)']:.1f}M params)")
    
    # Create comparison plots
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test accuracy comparison
        axes[0, 0].bar(df['Model'], df['Test Accuracy'])
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Model size comparison
        axes[0, 1].bar(df['Model'], df['Parameters (M)'])
        axes[0, 1].set_title('Model Size Comparison')
        axes[0, 1].set_ylabel('Parameters (M)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        axes[1, 0].bar(df['Model'], df['Training Time (min)'])
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Time (minutes)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy vs parameters scatter
        axes[1, 1].scatter(df['Parameters (M)'], df['Test Accuracy'])
        for _, row in df.iterrows():
            axes[1, 1].annotate(row['Model'], 
                               (row['Parameters (M)'], row['Test Accuracy']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Parameters (M)')
        axes[1, 1].set_ylabel('Test Accuracy')
        axes[1, 1].set_title('Efficiency: Accuracy vs Model Size')
        
        plt.tight_layout()
        plt.savefig('results/convnext_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Comparison plots saved to results/convnext_comparison.png")
        
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")


def main():
    """Main function to train ConvNeXt on CIFAR-100"""
    args = parse_args()
    
    print("=" * 80)
    print("CONVNEXT TRAINING ON CIFAR-100")
    print("=" * 80)
    
    if args.compare_all:
        # Compare all variants
        compare_all_variants(args)
    else:
        # Train single model
        results = train_single_model(args)
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED!")
        print("=" * 50)
        print(f"Model: ConvNeXt-{results['variant']} ({results['implementation']})")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
        print(f"Total Parameters: {results['total_params']:,}")
        print(f"Training Time: {results['training_time']:.2f} seconds")
        print("=" * 50)


if __name__ == "__main__":
    main()