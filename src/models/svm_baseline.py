# CIFAR-100 Image Classification Project - SVM Baseline with Feature Extraction

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import sys
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import seaborn as sns

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the correct path
from src.data.data_preparation import load_cifar100


def extract_features(dataloader, feature_extractor, device, batch_size=128, num_workers=4):
    """
    Extract features from a dataset using a pre-trained CNN
    
    Args:
        dataloader: DataLoader with the dataset
        feature_extractor: Pre-trained model to use as feature extractor
        device: Device to run extraction on (cuda/cpu)
        batch_size: Batch size for extraction
        num_workers: Number of workers for data loading
        
    Returns:
        features: Extracted features (N x D)
        labels: Corresponding labels (N)
    """
    feature_extractor.eval()
    
    # Lists to store features and labels
    all_features = []
    all_labels = []
    
    # Disable gradients
    with torch.no_grad():
        # Iterate over batches
        for inputs, labels in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            
            # Forward pass to extract features
            features = feature_extractor(inputs)
            
            # Store features and labels
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate batches
    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    
    return features, labels


def create_feature_extractor(model_name='resnet18'):
    """
    Create a feature extractor from a pre-trained model
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        feature_extractor: Feature extractor model
    """
    if model_name == 'resnet18':
        # Load pre-trained ResNet18
        try:
            # Try new PyTorch API first
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except (AttributeError, ImportError):
            # Fall back to older API
            model = models.resnet18(pretrained=True)
        
        # Remove the final fully-connected layer
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        
        # Add flatten layer to get a feature vector
        feature_extractor.add_module('flatten', torch.nn.Flatten())
        
    elif model_name == 'efficientnet_b0':
        # Load pre-trained EfficientNet B0
        try:
            # Try new PyTorch API first
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except (AttributeError, ImportError):
            # Fall back to older API
            model = models.efficientnet_b0(pretrained=True)
        
        # Remove the classifier
        model.classifier = torch.nn.Identity()
        feature_extractor = model
        
    else:
        raise ValueError(f"Feature extractor {model_name} not implemented")
    
    return feature_extractor


def train_and_evaluate_svm(train_loader, test_loader, classes, feature_extractor_name='resnet18'):
    """
    Train an SVM classifier using features extracted from a pre-trained CNN
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        classes: List of class names
        feature_extractor_name: Name of the model to use as feature extractor
        
    Returns:
        results: Dictionary with evaluation results
    """
    print(f"Creating feature extractor from {feature_extractor_name}...")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create feature extractor
    feature_extractor = create_feature_extractor(feature_extractor_name)
    feature_extractor = feature_extractor.to(device)
    
    # Create directories for results
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Check if features have been extracted before
    features_file = f"results/features_{feature_extractor_name}.pkl"
    
    if os.path.exists(features_file):
        print(f"Loading precomputed features from {features_file}...")
        with open(features_file, 'rb') as f:
            data = pickle.load(f)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        print("Extracting features from training set...")
        X_train, y_train = extract_features(train_loader, feature_extractor, device)
        
        print("Extracting features from test set...")
        X_test, y_test = extract_features(test_loader, feature_extractor, device)
        
        # Save extracted features to disk
        with open(features_file, 'wb') as f:
            pickle.dump({
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }, f)
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training SVM classifier on {X_train.shape[0]} samples...")
    model_name = f"svm_{feature_extractor_name}"
    
    # Check if SVM model already exists
    svm_model_file = f"models/{model_name}.pkl"
    
    if os.path.exists(svm_model_file):
        print(f"Loading trained SVM model from {svm_model_file}...")
        with open(svm_model_file, 'rb') as f:
            svm = pickle.load(f)
    else:
        # Train SVM (can take a while)
        start_time = time.time()
        
        # Using linear SVM for faster training
        svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovr', probability=True, verbose=True)
        svm.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"SVM trained in {training_time:.2f} seconds")
        
        # Save the model
        with open(svm_model_file, 'wb') as f:
            pickle.dump(svm, f)
    
    # Evaluate the model
    print("Evaluating SVM classifier...")
    start_time = time.time()
    
    # Predict on test set
    y_pred = svm.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    
    # Calculate weighted precision, recall, and F1 score
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Top-5 accuracy (try to get probabilities)
    try:
        # Get probability estimates for each class
        y_proba = svm.predict_proba(X_test)
        
        # Get top 5 predictions for each sample
        top5_indices = np.argsort(-y_proba, axis=1)[:, :5]
        
        # Check if true label is in top 5
        top5_correct = 0
        for i in range(len(y_test)):
            if y_test[i] in top5_indices[i]:
                top5_correct += 1
        
        top5_accuracy = top5_correct / len(y_test)
    except:
        # Fallback in case predict_proba is not available
        print("Warning: Unable to compute Top-5 accuracy for SVM")
        top5_accuracy = 0.0
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Evaluation time
    eval_time = time.time() - start_time
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Precision (Weighted): {weighted_precision:.4f}")
    print(f"Recall (Weighted): {weighted_recall:.4f}")
    print(f"F1 Score (Weighted): {weighted_f1:.4f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    
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
    
    # Store results
    results = {
        'model_name': model_name,
        'test_accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'confusion_matrix_norm': cm_norm,
        'all_preds': y_pred,
        'all_labels': y_test,
        'training_time': [0]  # Placeholder for compatibility with other models
    }
    
    return results


if __name__ == "__main__":
    # Load the data
    batch_size = 64  # Smaller batch size for feature extraction
    train_loader, test_loader, classes = load_cifar100(batch_size=batch_size)
    
    # Train and evaluate SVM with features extracted from ResNet18
    results = train_and_evaluate_svm(train_loader, test_loader, classes, 
                                     feature_extractor_name='resnet18')
    
    # Save results to file
    with open(f"results/{results['model_name']}_results.pkl", 'wb') as f:
        pickle.dump(results, f)