#!/usr/bin/env python3
# scripts/train_vision_transformer.py
# CIFAR-100 Image Classification Project - Vision Transformer Training Script (Fixed)

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import with proper error handling
try:
    from src.data.data_preparation import load_cifar100
except ImportError:
    try:
        sys.path.append(os.path.join(project_root, 'src'))
        from src.data.data_preparation import load_cifar100
    except ImportError:
        print("Error: Could not import data_preparation. Please check your project structure.")
        sys.exit(1)

try:
    from src.models.pretrained_models import get_model
except ImportError:
    try:
        sys.path.append(os.path.join(project_root, 'src'))
        from src.models.pretrained_models import get_model
    except ImportError:
        print("Error: Could not import get_model. Please check your project structure.")
        sys.exit(1)

# Standalone functions - no external imports needed
def evaluate_model_simple(model, test_loader, criterion, device, classes, model_name):
    """Simple evaluation function"""
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f"\nTest Results for {model_name}:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {avg_loss:.4f}")
    
    return {
        'test_accuracy': accuracy, 
        'test_loss': avg_loss,
        'model_name': model_name
    }

def plot_training_history_simple(history, model_name, save_dir='results'):
    """Simple plotting function"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train', linewidth=2)
    plt.plot(history['val_acc'], label='Val', linewidth=2)
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Val', linewidth=2)
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rates'], linewidth=2, color='orange')
    plt.title(f'{model_name} - Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_vit_configs():
    """Create different ViT configurations for experimentation"""
    configs = {
        'vit_pretrained': {
            'model_name': 'vit',
            'description': 'Pretrained ViT Base (torchvision)',
            'learning_rate': 0.0001,
            'batch_size': 64,
            'weight_decay': 1e-5,
            'warmup_epochs': 3,
        },
        'resnet18_baseline': {
            'model_name': 'resnet18',
            'description': 'ResNet18 Baseline',
            'learning_rate': 0.001,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'warmup_epochs': 5,
        }
    }
    
    # Try to add custom ViT models if available
    try:
        from src.models.pretrained_models import list_available_models
        available = list_available_models()
        if 'vit_tiny' in available:
            configs['vit_tiny_custom'] = {
                'model_name': 'vit_tiny',
                'description': 'Custom ViT Tiny (192 dim, 3 heads)',
                'learning_rate': 0.001,
                'batch_size': 128,
                'weight_decay': 1e-4,
                'warmup_epochs': 5,
            }
        if 'vit_small' in available:
            configs['vit_small_custom'] = {
                'model_name': 'vit_small',
                'description': 'Custom ViT Small (384 dim, 6 heads)',
                'learning_rate': 0.0008,
                'batch_size': 64,
                'weight_decay': 1e-4,
                'warmup_epochs': 5,
            }
    except:
        pass
    
    return configs


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


def train_model_complete(config, train_loader, val_loader, test_loader, classes, device, 
                        num_epochs=50, patience=10, save_dir='results'):
    """Complete training function with all functionality built-in"""
    
    model_name = config['model_name']
    print(f"\n{'='*60}")
    print(f"Training {config['description']}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # Create model
    try:
        model = get_model(model_name, num_classes=100, pretrained=('pretrained' in config['description']))
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        print("Falling back to ResNet18...")
        model = get_model('resnet18', num_classes=100, pretrained=True)
        model_name = 'resnet18_fallback'
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer
    if 'pretrained' in config['description'].lower():
        # Lower learning rate for pretrained models
        optimizer = optim.AdamW(model.parameters(), 
                              lr=config['learning_rate'] * 0.1, 
                              weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), 
                              lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=config['warmup_epochs'],
        total_epochs=num_epochs,
        base_lr=config['learning_rate']
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'time_per_epoch': []
    }
    
    # Early stopping variables
    best_val_acc = 0.0
    patience_counter = 0
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Update learning rate
        current_lr = scheduler.step()
        history['learning_rates'].append(current_lr)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': train_loss / (batch_idx + 1),
                'Acc': 100. * train_correct / train_total,
                'LR': f'{current_lr:.6f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch_idx, (inputs, targets) in enumerate(val_pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': val_loss / (batch_idx + 1),
                    'Acc': 100. * val_correct / val_total
                })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['time_per_epoch'].append(epoch_time)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}")
        print(f"Val   - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
        print(f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'config': config,
                'history': history
            }, f"models/{model_name}_best.pth")
            print(f"✓ Saved best model with validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 60)
    
    # Load best model for evaluation
    try:
        checkpoint = torch.load(f"models/{model_name}_best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded best model for evaluation")
    except:
        print("⚠ Warning: Could not load best model, using current model state")
    
    # Evaluate on test set
    print(f"\nEvaluating {model_name} on test set...")
    test_results = evaluate_model_simple(model, test_loader, criterion, device, classes, model_name)
    
    # Plot training history
    plot_training_history_simple(history, model_name, save_dir)
    
    # Combine results
    results = {
        'model_name': model_name,
        'config': config,
        'history': history,
        'test_results': test_results,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    return results


def main():
    """Main function to train Vision Transformer models"""
    parser = argparse.ArgumentParser(description='Train Vision Transformer models on CIFAR-100')
    parser.add_argument('--config', type=str, default='vit_pretrained', 
                       help='Which configuration to run')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='results/vit_experiments', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    try:
        train_loader, test_loader, classes = load_cifar100(batch_size=128)
        print(f"✓ Data loaded: {len(classes)} classes")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Create train/validation split
    train_size = int(0.9 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {train_size} train, {val_size} val, {len(test_loader.dataset)} test")
    
    # Get configurations
    configs = create_vit_configs()
    print(f"Available configurations: {list(configs.keys())}")
    
    # Select configuration
    if args.config == 'all':
        configs_to_run = configs
    else:
        if args.config not in configs:
            print(f"Configuration '{args.config}' not found. Available: {list(configs.keys())}")
            print("Using default 'vit_pretrained' configuration")
            configs_to_run = {'vit_pretrained': configs['vit_pretrained']}
        else:
            configs_to_run = {args.config: configs[args.config]}
    
    all_results = []
    
    # Train models
    for config_name, config in configs_to_run.items():
        print(f"\n🚀 Starting training for configuration: {config_name}")
        
        try:
            # Create data loaders with config-specific batch size
            train_loader_config = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
            
            val_loader_config = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
            
            test_loader_config = torch.utils.data.DataLoader(
                test_loader.dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
            
            # Train the model
            results = train_model_complete(
                config, 
                train_loader_config, 
                val_loader_config, 
                test_loader_config,
                classes, 
                device, 
                num_epochs=args.epochs, 
                patience=args.patience,
                save_dir=args.save_dir
            )
            all_results.append(results)
            
            print(f"✅ Training completed for {config_name}")
            
        except Exception as e:
            print(f"❌ Error training {config_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if len(all_results) > 0:
        print(f"\n{'='*60}")
        print("🎉 TRAINING COMPLETED!")
        print(f"{'='*60}")
        
        for result in all_results:
            print(f"\n📊 {result['model_name']}:")
            print(f"  Test Accuracy: {result['test_results']['test_accuracy']:.4f}")
            print(f"  Parameters: {result['total_params']:,}")
            print(f"  Avg. Training Time: {np.mean(result['history']['time_per_epoch']):.2f}s/epoch")
            print(f"  Best Val Accuracy: {max(result['history']['val_acc']):.4f}")
        
        print(f"\n📁 All results saved to: {args.save_dir}")
        print("📈 Check the training history plots in the results folder")
        print("💾 Trained models saved in the 'models' folder")
    else:
        print("❌ No models were trained successfully.")
        print("\nTroubleshooting tips:")
        print("1. Check if the fixed pretrained_models.py file is in place")
        print("2. Verify your Python environment has all dependencies")
        print("3. Try running with a simpler configuration: --config resnet18_baseline")


if __name__ == "__main__":
    main()