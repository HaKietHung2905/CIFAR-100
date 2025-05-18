#!/usr/bin/env python
# CIFAR-100 Image Classification Project - ResNet Model Comparison

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import yaml

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project modules
from src.models.pretrained_models import ResNet18Model, ResNet50Model
from src.models.resnet34 import create_resnet34
from src.data.data_preparation import load_cifar100
from scripts.train_resnet_cifar100 import create_model, load_config


def load_results(model_names):
    """
    Load results for the specified models
    
    Args:
        model_names: List of model names
        
    Returns:
        results_list: List of result dictionaries
    """
    results_list = []
    
    for model_name in model_names:
        results_path = f"results/model_analysis/{model_name}_results.pkl"
        
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            results_list.append(results)
            print(f"Loaded results for {model_name}")
        else:
            print(f"Warning: Results not found for {model_name}")
    
    return results_list


def compare_models(results_list, model_names):
    """
    Compare models based on their evaluation results
    
    Args:
        results_list: List of result dictionaries
        model_names: List of model names
        
    Returns:
        comparison_df: DataFrame with model comparison
    """
    # Check if we have results to compare
    if not results_list:
        print("No results to compare")
        return None
    
    # Create a DataFrame with the results
    comparison_data = []
    
    for i, results in enumerate(results_list):
        model_name = model_names[i] if i < len(model_names) else results.get('model_name', f"Model_{i}")
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results.get('test_accuracy', 0),
            'Top-5 Accuracy': results.get('top5_accuracy', 0),
            'F1 Score': results.get('f1', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by accuracy
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save to CSV
    os.makedirs('results/comparison', exist_ok=True)
    comparison_df.to_csv('results/comparison/model_comparison.csv', index=False)
    
    return comparison_df


def plot_comparison(comparison_df):
    """
    Plot model comparison
    
    Args:
        comparison_df: DataFrame with model comparison
    """
    if comparison_df is None or len(comparison_df) == 0:
        return
    
    # Create directories for plots
    os.makedirs('results/comparison', exist_ok=True)
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar plot
    metrics = ['Accuracy', 'Top-5 Accuracy', 'F1 Score']
    x = np.arange(len(comparison_df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        bars = plt.bar(x + offset, comparison_df[metric], width, label=metric)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add labels and legend
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Model Comparison', fontsize=16)
    plt.xticks(x, comparison_df['Model'])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/comparison/model_comparison.png', dpi=200)
    plt.close()
    
    # Create a radar plot
    plt.figure(figsize=(10, 8))
    
    # Prepare data for radar plot
    categories = ['Accuracy', 'Top-5 Accuracy', 'F1 Score', 'Precision', 'Recall']
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Plot each model
    for i, row in comparison_df.iterrows():
        values = [row[cat] for cat in categories]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, label=row['Model'])
        ax.fill(angles, values, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Comparison - Radar Plot', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/comparison/model_comparison_radar.png', dpi=200)
    plt.close()


def model_parameter_comparison(model_names):
    """
    Compare model parameters and complexity
    
    Args:
        model_names: List of model names
    """
    # Create models
    models = []
    model_depths = []
    
    for model_name in model_names:
        if 'resnet18' in model_name:
            models.append(ResNet18Model(num_classes=100))
            model_depths.append(18)
        elif 'resnet34' in model_name:
            models.append(create_resnet34(num_classes=100))
            model_depths.append(34)
        elif 'resnet50' in model_name:
            models.append(ResNet50Model(num_classes=100))
            model_depths.append(50)
    
    # Calculate parameters
    param_counts = [sum(p.numel() for p in model.parameters()) for model in models]
    
    # Create DataFrame
    param_df = pd.DataFrame({
        'Model': model_names,
        'Depth': model_depths,
        'Parameters': param_counts,
        'Parameters (M)': [count / 1_000_000 for count in param_counts]
    })
    
    # Print table
    print("\nModel Parameter Comparison:")
    print(param_df.to_string(index=False))
    
    # Save to CSV
    param_df.to_csv('results/comparison/model_parameters.csv', index=False)
    
    # Plot parameter comparison
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(param_df['Model'], param_df['Parameters (M)'])
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}M', ha='center', va='bottom')
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Parameters (Millions)', fontsize=14)
    plt.title('Model Parameter Comparison', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/comparison/model_parameters.png', dpi=200)
    plt.close()
    
    return param_df


def analyze_class_performance(results_list, model_names, classes):
    """
    Analyze and compare per-class performance across models
    
    Args:
        results_list: List of result dictionaries
        model_names: List of model names
        classes: List of class names
    """
    if not results_list:
        return
    
    # Create a list to store per-class metrics for each model
    model_class_metrics = []
    
    for i, results in enumerate(results_list):
        model_name = model_names[i] if i < len(model_names) else results.get('model_name', f"Model_{i}")
        
        # Check if we have per-class metrics
        if 'per_class_f1' not in results:
            print(f"Warning: No per-class metrics found for {model_name}")
            continue
        
        # Create DataFrame with per-class metrics
        df = pd.DataFrame({
            'Class': classes,
            'F1 Score': results['per_class_f1'],
            'Precision': results['per_class_precision'],
            'Recall': results['per_class_recall'],
            'Support': results['per_class_support']
        })
        
        df['Model'] = model_name
        model_class_metrics.append(df)
    
    # Combine all DataFrames
    if model_class_metrics:
        all_metrics = pd.concat(model_class_metrics, ignore_index=True)
        
        # Find the most challenging classes (lowest average F1 score across models)
        avg_f1_by_class = all_metrics.groupby('Class')['F1 Score'].mean().reset_index()
        challenging_classes = avg_f1_by_class.sort_values('F1 Score').head(10)['Class'].tolist()
        
        # Compare models on challenging classes
        challenging_df = all_metrics[all_metrics['Class'].isin(challenging_classes)]
        
        # Plot comparison for challenging classes
        plt.figure(figsize=(14, 10))
        
        # Create grouped bar plot
        sns.barplot(x='Class', y='F1 Score', hue='Model', data=challenging_df)
        
        plt.xlabel('Class', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.title('Model Comparison on Challenging Classes', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend(title='Model')
        
        plt.tight_layout()
        plt.savefig('results/comparison/challenging_classes_comparison.png', dpi=200)
        plt.close()
        
        return challenging_df
    
    return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare ResNet models on CIFAR-100')
    parser.add_argument('--models', nargs='+', default=['resnet18_pretrained', 'resnet34_pretrained', 'resnet50_pretrained'],
                        help='Models to compare')
    parser.add_argument('--metric', type=str, default='Accuracy',
                        help='Primary metric for comparison')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("CIFAR-100 Image Classification - ResNet Model Comparison")
    print("=" * 80)
    
    # Load results
    results_list = load_results(args.models)
    
    # Compare models
    comparison_df = compare_models(results_list, args.models)
    
    # Plot comparison
    if comparison_df is not None:
        plot_comparison(comparison_df)
    
    # Compare model parameters
    param_df = model_parameter_comparison(args.models)
    
    # Load class names
    _, _, _, classes = load_cifar100(batch_size=1)
    
    # Analyze class performance
    challenging_df = analyze_class_performance(results_list, args.models, classes)
    
    # Print best model
    if comparison_df is not None and not comparison_df.empty:
        best_model_idx = comparison_df[args.metric].idxmax()
        best_model = comparison_df.loc[best_model_idx, 'Model']
        best_acc = comparison_df.loc[best_model_idx, args.metric]
        
        print(f"\nBest model by {args.metric}: {best_model} ({best_acc:.4f})")
    
    print("\nComparison completed successfully!")


if __name__ == "__main__":
    main()