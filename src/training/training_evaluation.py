# CIFAR-100 Image Classification Project - Training and Evaluation

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import seaborn as sns
import os

from pretrained_models import get_model
from data_preparation import load_cifar100

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_epochs=30, patience=5, model_name="model"):
    """
    Train a model and evaluate on validation set
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        num_epochs: Maximum number of epochs to train
        patience: Early stopping patience
        model_name: Name for saving the model
        
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    # Initialize best validation accuracy and patience counter
    best_val_acc = 0.0
    patience_counter = 0
    
    # History for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'time_per_epoch': []
    }
    
    print(f"Training {model_name}...")
    
    for epoch in range(num_epochs):
        # Start time for epoch
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in train_pbar:
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
            total += labels.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': running_loss / total, 
                'acc': running_corrects / total
            })
        
        # Calculate epoch metrics
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        # Disable gradients during validation
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': running_loss / len(val_loader.dataset), 
                    'acc': running_corrects / len(val_loader.dataset)
                })
        
        # Calculate epoch metrics
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects / len(val_loader.dataset)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
        
        # Calculate time per epoch
        epoch_time = time.time() - epoch_start
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['time_per_epoch'].append(epoch_time)
        
        # Check if this is the best model so far
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            
            # Save the best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            print(f"Saved best model with accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f"models/{model_name}_best.pth"))
    
    return model, history


def evaluate_model(model, test_loader, criterion, device, classes, model_name="model"):
    """
    Evaluate the model on the test set
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        classes: List of class names
        model_name: Name of the model
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    model.eval()
    
    # Lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    
    # Running statistics
    running_loss = 0.0
    running_corrects = 0
    
    # Disable gradients during evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
    
    # Calculate overall metrics
    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = running_corrects / len(test_loader.dataset)
    
    # Calculate top-5 accuracy
    topk_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get top-5 predictions
            _, topk_preds = torch.topk(outputs, k=5, dim=1)
            
            # Check if true label is in top-5 predictions
            for i in range(len(labels)):
                if labels[i] in topk_preds[i]:
                    topk_correct += 1
    
    top5_accuracy = topk_correct / len(test_loader.dataset)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    # Calculate weighted precision, recall, and F1 score
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Store all results
    results = {
        'model_name': model_name,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'confusion_matrix_norm': cm_norm,
        'all_preds': all_preds,
        'all_labels': all_labels
    }
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy (Top-1): {test_accuracy:.4f}")
    print(f"Test Accuracy (Top-5): {top5_accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Precision (Weighted): {weighted_precision:.4f}")
    print(f"Recall (Weighted): {weighted_recall:.4f}")
    print(f"F1 Score (Weighted): {weighted_f1:.4f}")
    
    # Visualize confusion matrix (sample of 20 classes for readability)
    plt.figure(figsize=(12, 10))
    # Select a subset of classes for better visualization
    num_classes_to_show = min(20, len(classes))
    
    # Get indices of the most confused classes (those with lowest diagonal values)
    diag_indices = np.diag_indices(len(cm_norm))
    diag_values = cm_norm[diag_indices]
    worst_class_indices = np.argsort(diag_values)[:num_classes_to_show]
    
    # Create a subset confusion matrix
    cm_subset = cm_norm[worst_class_indices][:, worst_class_indices]
    classes_subset = [classes[i] for i in worst_class_indices]
    
    sns.heatmap(cm_subset, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes_subset, yticklabels=classes_subset)
    plt.title(f'Confusion Matrix (Top {num_classes_to_show} Most Confused Classes) - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_{model_name}.png")
    
    # Plot misclassified examples
    # Get indices of misclassified samples
    misclassified_indices = np.where(np.array(all_preds) != np.array(all_labels))[0]
    
    # If there are misclassified samples, plot some examples
    if len(misclassified_indices) > 0:
        # Get a batch of test data
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        
        # Create a figure to show misclassified images
        plt.figure(figsize=(12, 8))
        
        for i, idx in enumerate(misclassified_indices[:min(10, len(misclassified_indices))]):
            # Find batch and index within batch
            batch_idx = idx // test_loader.batch_size
            idx_in_batch = idx % test_loader.batch_size
            
            # Skip if batch is not the current one
            if batch_idx != 0:
                continue
                
            # Plot image
            if idx_in_batch < len(images):
                plt.subplot(2, 5, i + 1)
                img = images[idx_in_batch].cpu().numpy().transpose((1, 2, 0))
                # Denormalize image
                mean = np.array([0.5071, 0.4867, 0.4408])
                std = np.array([0.2675, 0.2565, 0.2761])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                plt.imshow(img)
                plt.title(f"True: {classes[labels[idx_in_batch]]}\nPred: {classes[all_preds[idx]]}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"results/misclassified_{model_name}.png")
    
    return results


def plot_training_history(history, model_name):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        model_name: Name of the model
    """
    # Create directory for results
    os.makedirs('results', exist_ok=True)
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/training_history_{model_name}.png")
    
    # Plot training time per epoch
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(history['time_per_epoch']) + 1), history['time_per_epoch'])
    plt.title(f'{model_name} - Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(f"results/training_time_{model_name}.png")


def compare_models(results_list):
    """
    Compare multiple models
    
    Args:
        results_list: List of dictionaries with evaluation results
    """
    # Create a DataFrame with the results
    df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy (Top-1)': r['test_accuracy'],
            'Accuracy (Top-5)': r['top5_accuracy'],
            'F1 Score (Macro)': r['f1'],
            'Precision (Macro)': r['precision'],
            'Recall (Macro)': r['recall'],
            'Training Time (s)': np.mean(r.get('training_time', [0]))
        }
        for r in results_list
    ])
    
    # Sort by accuracy
    df = df.sort_values('Accuracy (Top-1)', ascending=False)
    
    # Save results to CSV
    df.to_csv('results/model_comparison.csv', index=False)
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(df['Model'], df['Accuracy (Top-1)'])
    plt.title('Top-1 Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    # Top-5 Accuracy comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(df['Model'], df['Accuracy (Top-5)'])
    plt.title('Top-5 Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    # F1 Score comparison
    plt.subplot(2, 2, 3)
    bars = plt.bar(df['Model'], df['F1 Score (Macro)'])
    plt.title('F1 Score (Macro)')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    # Training Time comparison
    plt.subplot(2, 2, 4)
    bars = plt.bar(df['Model'], df['Training Time (s)'])
    plt.title('Average Training Time per Epoch')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1f}s', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    
    # Create a radar plot for comprehensive comparison
    plt.figure(figsize=(10, 8))
    
    # Prepare the data for radar plot
    metrics = ['Accuracy (Top-1)', 'Accuracy (Top-5)', 'F1 Score (Macro)', 
               'Precision (Macro)', 'Recall (Macro)']
    
    # Number of variables
    N = len(metrics)
    
    # Create a list of angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Add each model to the radar plot
    for i, row in df.iterrows():
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Plot the model
        ax.plot(angles, values, linewidth=2, label=row['Model'])
        ax.fill(angles, values, alpha=0.1)
    
    # Set the angle labels
    plt.xticks(angles[:-1], metrics)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Comparison - Radar Plot')
    plt.tight_layout()
    plt.savefig('results/radar_comparison.png')
    
    return df


def main():
    """Main function to run the training and evaluation"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories for results and models
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load the data
    batch_size = 128
    train_loader, test_loader, classes = load_cifar100(batch_size=batch_size)
    
    # Split training data into training and validation
    train_size = int(0.8 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Define models to train
    model_names = [
        'custom_cnn',
        'resnet18',
        'vgg16',
        'densenet121',
        'efficientnet_b0',
        'convnext',
        'vit',
        'swin'
    ]
    
    # Store results for comparison
    all_results = []
    
    # Train and evaluate each model
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Training and evaluating model: {model_name}")
        print(f"{'='*50}")
        
        # Get model
        model = get_model(model_name, num_classes=100, pretrained=True)
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        
        # Train the model
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, num_epochs=30, patience=5, model_name=model_name
        )
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Evaluate the model
        results = evaluate_model(model, test_loader, criterion, device, classes, model_name=model_name)
        
        # Add training time to results
        results['training_time'] = history['time_per_epoch']
        
        # Store results
        all_results.append(results)
        
        # Save detailed results to CSV
        # Create a DataFrame with per-class metrics
        class_results = pd.DataFrame({
            'Class': classes,
            'Precision': precision_recall_fscore_support(results['all_labels'], results['all_preds'], average=None)[0],
            'Recall': precision_recall_fscore_support(results['all_labels'], results['all_preds'], average=None)[1],
            'F1 Score': precision_recall_fscore_support(results['all_labels'], results['all_preds'], average=None)[2],
            'Support': precision_recall_fscore_support(results['all_labels'], results['all_preds'], average=None)[3]
        })
        
        # Sort by F1 Score (ascending) to identify problematic classes
        class_results = class_results.sort_values('F1 Score')
        
        # Save to CSV
        class_results.to_csv(f"results/{model_name}_class_metrics.csv", index=False)
        
        # Free up memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    # Compare all models
    comparison_df = compare_models(all_results)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()