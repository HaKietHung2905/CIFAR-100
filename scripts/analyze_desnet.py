#!/usr/bin/env python
# CIFAR-100 Image Classification Project - DenseNet121 Analysis

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import argparse
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import cv2

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project modules
from src.data.data_preparation import load_cifar100
from src.models.pretrained_models import DenseNet121Model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze DenseNet121 performance on CIFAR-100')
    parser.add_argument('--model-path', type=str, default='./models/densenet121_best.pth', 
                        help='Path to trained model')
    parser.add_argument('--results-path', type=str, default='./results/model_analysis/densenet121_results.pkl',
                        help='Path to training results')
    parser.add_argument('--output-dir', type=str, default='./results/densenet121_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def load_model_and_results(model_path, results_path, device):
    """
    Load trained model and results
    
    Args:
        model_path: Path to model weights
        results_path: Path to results file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        results: Results dictionary (if available)
    """
    # Load model
    model = DenseNet121Model(num_classes=100)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Using random weights for analysis")
    
    model = model.to(device)
    model.eval()
    
    # Load results if available
    results = None
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded results from {results_path}")
    else:
        print(f"Warning: Results not found at {results_path}")
    
    return model, results

def evaluate_model(model, test_loader, device, classes):
    """
    Evaluate model performance on test data
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run inference on
        classes: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Disable gradients during evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating DenseNet121"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions, probabilities and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Calculate top-1 accuracy
    accuracy = np.mean(all_preds == all_labels)
    
    # Calculate top-5 accuracy
    top5_correct = 0
    for i in range(len(all_labels)):
        # Get indices of top 5 predictions
        topk_indices = np.argsort(all_probs[i])[-5:]
        if all_labels[i] in topk_indices:
            top5_correct += 1
    
    top5_accuracy = top5_correct / len(all_labels)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # Calculate weighted precision, recall, and F1 score
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Print results
    print(f"\nDenseNet121 Evaluation Results:")
    print(f"Top-1 Accuracy: {accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1 Score (Weighted): {weighted_f1:.4f}")
    
    # Store results
    results = {
        'model_name': 'densenet121',
        'test_accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }
    
    return results

def analyze_superclass_performance(results, classes, output_dir):
    """
    Analyze model performance by CIFAR-100 superclasses
    
    Args:
        results: Results dictionary from evaluation
        classes: List of class names
        output_dir: Directory to save analysis
    """
    os.makedirs(os.path.join(output_dir, 'superclass_analysis'), exist_ok=True)
    
    # Define CIFAR-100 superclasses
    superclasses = {
        'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
        'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
        'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
        'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
        'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
        'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
        'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
        'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
        'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
        'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
        'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
        'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
        'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
        'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
        'people': ['baby', 'boy', 'girl', 'man', 'woman'],
        'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
        'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
        'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
        'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    }
    
    # Create mapping from class name to index
    class_to_idx = {name: i for i, name in enumerate(classes)}
    
    # Get predictions and labels
    all_preds = results['all_preds']
    all_labels = results['all_labels']
    
    # Calculate per-superclass metrics
    superclass_metrics = {}
    
    for superclass in superclasses:
        # Get indices of classes in this superclass
        class_indices = [class_to_idx[name] for name in superclasses[superclass] if name in class_to_idx]
        
        # Get samples of this superclass
        mask = np.isin(all_labels, class_indices)
        
        if np.sum(mask) > 0:
            # Calculate accuracy
            accuracy = np.mean(all_preds[mask] == all_labels[mask])
            
            # Calculate precision, recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[mask], all_preds[mask], average='macro'
            )
            
            # Store metrics
            superclass_metrics[superclass] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_samples': np.sum(mask)
            }
    
    # Create DataFrame
    superclass_df = pd.DataFrame.from_dict(superclass_metrics, orient='index')
    superclass_df = superclass_df.sort_values('accuracy', ascending=False)
    
    # Save to CSV
    superclass_df.to_csv(os.path.join(output_dir, 'superclass_analysis', 'superclass_metrics.csv'))
    
    # Plot superclass performance
    plt.figure(figsize=(14, 8))
    sns.barplot(x=superclass_df.index, y=superclass_df['accuracy'])
    plt.xticks(rotation=90)
    plt.title('DenseNet121 - Accuracy by Superclass')
    plt.ylabel('Accuracy')
    plt.xlabel('Superclass')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'superclass_analysis', 'superclass_accuracy.png'))
    plt.close()
    
    # Plot F1 scores by superclass
    plt.figure(figsize=(14, 8))
    sns.barplot(x=superclass_df.index, y=superclass_df['f1'])
    plt.xticks(rotation=90)
    plt.title('DenseNet121 - F1 Score by Superclass')
    plt.ylabel('F1 Score')
    plt.xlabel('Superclass')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'superclass_analysis', 'superclass_f1.png'))
    plt.close()
    
    print(f"Superclass analysis saved to {os.path.join(output_dir, 'superclass_analysis')}")
    
    return superclass_df

def analyze_challenging_classes(results, classes, output_dir):
    """
    Analyze the most challenging classes for the model
    
    Args:
        results: Results dictionary from evaluation
        classes: List of class names
        output_dir: Directory to save analysis
    """
    os.makedirs(os.path.join(output_dir, 'class_analysis'), exist_ok=True)
    
    # Get per-class F1 scores
    per_class_f1 = results['per_class_f1']
    per_class_precision = results['per_class_precision']
    per_class_recall = results['per_class_recall']
    per_class_support = results['per_class_support']
    
    # Create DataFrame with per-class metrics
    class_df = pd.DataFrame({
        'Class': classes,
        'F1 Score': per_class_f1,
        'Precision': per_class_precision,
        'Recall': per_class_recall,
        'Support': per_class_support
    })
    
    # Sort by F1 Score (ascending) to identify challenging classes
    class_df = class_df.sort_values('F1 Score')
    
    # Save to CSV
    class_df.to_csv(os.path.join(output_dir, 'class_analysis', 'per_class_metrics.csv'), index=False)
    
    # Plot worst performing classes
    worst_classes = class_df.head(20)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=worst_classes, x='Class', y='F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.title('DenseNet121 - 20 Most Challenging Classes')
    plt.ylabel('F1 Score')
    plt.xlabel('Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_analysis', 'worst_classes.png'))
    plt.close()
    
    # Plot best performing classes
    best_classes = class_df.tail(20)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=best_classes, x='Class', y='F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.title('DenseNet121 - 20 Best Performing Classes')
    plt.ylabel('F1 Score')
    plt.xlabel('Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_analysis', 'best_classes.png'))
    plt.close()
    
    print(f"Class analysis saved to {os.path.join(output_dir, 'class_analysis')}")
    
    return class_df

def analyze_confusion_patterns(results, classes, output_dir):
    """
    Analyze confusion patterns in predictions
    
    Args:
        results: Results dictionary from evaluation
        classes: List of class names
        output_dir: Directory to save analysis
    """
    os.makedirs(os.path.join(output_dir, 'confusion_analysis'), exist_ok=True)
    
    all_preds = results['all_preds']
    all_labels = results['all_labels']
    
    # Find misclassified samples
    misclassified_mask = all_preds != all_labels
    misclassified_true = all_labels[misclassified_mask]
    misclassified_pred = all_preds[misclassified_mask]
    
    # Count confusion pairs
    confusion_pairs = {}
    for true_label, pred_label in zip(misclassified_true, misclassified_pred):
        pair = (classes[true_label], classes[pred_label])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    # Get most common confusion pairs
    most_common_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Create DataFrame
    confusion_df = pd.DataFrame([
        {
            'True Class': pair[0],
            'Predicted Class': pair[1],
            'Count': count
        }
        for (pair, count) in most_common_pairs
    ])
    
    # Save to CSV
    confusion_df.to_csv(os.path.join(output_dir, 'confusion_analysis', 'confusion_pairs.csv'), index=False)
    
    # Plot most common confusions
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(confusion_df))
    labels = [f"{row['True Class']} → {row['Predicted Class']}" for _, row in confusion_df.iterrows()]
    
    plt.barh(y_pos, confusion_df['Count'])
    plt.yticks(y_pos, labels)
    plt.xlabel('Number of Misclassifications')
    plt.title('DenseNet121 - Most Common Confusion Pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_analysis', 'confusion_pairs.png'))
    plt.close()
    
    print(f"Confusion analysis saved to {os.path.join(output_dir, 'confusion_analysis')}")
    
    return confusion_df

def save_analysis_results(results, output_dir):
    """
    Save complete analysis results
    
    Args:
        results: Results dictionary
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    with open(os.path.join(output_dir, 'densenet121_analysis_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Create summary report
    summary = {
        'Model': 'DenseNet121',
        'Test Accuracy': results['test_accuracy'],
        'Top-5 Accuracy': results['top5_accuracy'],
        'F1 Score (Macro)': results['f1'],
        'Precision (Macro)': results['precision'],
        'Recall (Macro)': results['recall'],
        'F1 Score (Weighted)': results['weighted_f1']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    
    print(f"Analysis results saved to {output_dir}")

def main():
    """Main function to analyze DenseNet121 performance"""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 80)
    print("CIFAR-100 Image Classification - DenseNet121 Analysis")
    print("=" * 80)
    
    # Load CIFAR-100 dataset
    _, test_loader, classes = load_cifar100(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Load model and existing results
    model, existing_results = load_model_and_results(args.model_path, args.results_path, device)
    
    # Evaluate model (if we don't have existing results or want fresh evaluation)
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, test_loader, device, classes)
    
    # Analyze superclass performance
    print("\nAnalyzing superclass performance...")
    superclass_df = analyze_superclass_performance(results, classes, args.output_dir)
    
    # Analyze challenging classes
    print("\nAnalyzing challenging classes...")
    class_df = analyze_challenging_classes(results, classes, args.output_dir)
    
    # Analyze confusion patterns
    print("\nAnalyzing confusion patterns...")
    confusion_df = analyze_confusion_patterns(results, classes, args.output_dir)
    
    # Save all results
    save_analysis_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Model: DenseNet121")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['f1']:.4f}")
    print(f"Best Superclass: {superclass_df.index[0]} ({superclass_df.iloc[0]['accuracy']:.4f})")
    print(f"Worst Superclass: {superclass_df.index[-1]} ({superclass_df.iloc[-1]['accuracy']:.4f})")
    print(f"Most Challenging Class: {class_df.iloc[0]['Class']} (F1: {class_df.iloc[0]['F1 Score']:.4f})")
    print(f"Best Performing Class: {class_df.iloc[-1]['Class']} (F1: {class_df.iloc[-1]['F1 Score']:.4f})")
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()