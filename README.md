# CIFAR-100 Image Classification Project

A comprehensive deep learning project for image classification on the CIFAR-100 dataset, implementing and comparing various state-of-the-art neural network architectures.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Models](#available-models)
- [Training Models](#training-models)
- [Evaluation and Analysis](#evaluation-and-analysis)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides a complete framework for training and evaluating deep learning models on the CIFAR-100 dataset. CIFAR-100 consists of 60,000 32x32 color images in 100 classes, with 600 images per class (500 training + 100 test images per class). The 100 classes are grouped into 20 superclasses.

## ✨ Features

- **Multiple Architecture Support**: Custom CNN, ResNet, VGG, DenseNet, EfficientNet, ConvNeXt, Vision Transformer, Swin Transformer
- **Advanced Training Techniques**: Data augmentation, mixup, cutmix, label smoothing, learning rate scheduling
- **Comprehensive Evaluation**: Top-1/Top-5 accuracy, confusion matrices, per-class metrics, superclass analysis
- **SVM Baseline**: Feature extraction with pre-trained CNNs + SVM classification
- **Visualization Tools**: Training curves, confusion matrices, misclassified samples, feature maps
- **Experiment Tracking**: Configurable logging, model checkpointing, results comparison
- **Easy Configuration**: YAML-based configuration files for different models

## 📁 Project Structure

```
CIFAR100_Classification/
├── configs/                    # Configuration files
│   ├── custom_cnn_config.yaml
│   ├── resnet_config.yaml
│   ├── swin_transformer_config.yaml
│   └── convnext_config.yaml
├── src/                        # Source code
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_preparation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── custom_cnn.py
│   │   ├── pretrained_models.py
│   │   └── svm_baseline.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── visualization_analysis.py
│   └── training/
│       ├── __init__.py
│       └── training_evaluation.py
├── scripts/                    # Training and analysis scripts
│   ├── setup_environment.py
│   ├── download_dataset.py
│   ├── train_custom_cnn.py
│   ├── train_resnet_cifar100.py
│   ├── train_swin_transformer.py
│   ├── train_vision_transformer.py
│   ├── train_efficientnet.py
│   ├── train_convnext.py
│   ├── train_densenet.py
│   ├── analyze_resnet.py
│   ├── analyze_densenet.py
│   └── compare_resnet.py
├── notebooks/                  # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── requirements.txt           # Python dependencies
├── create_notebook.py        # Notebook generation script
└── README.md                 # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Method 1: Automatic Setup

Run the setup script to automatically install dependencies and create necessary directories:

```bash
# Clone the repository
git clone <repository-url>
cd CIFAR100_Classification

# Run setup script
python scripts/setup_environment.py

# For conda users
python scripts/setup_environment.py --conda --env-name cifar100
```

### Method 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn
pip install opencv-python albumentations
pip install tqdm pyyaml

# Create necessary directories
mkdir -p data models results/{plots,model_analysis,confusion_matrices}
```

### Download Dataset

The CIFAR-100 dataset will be automatically downloaded when you first run a training script. Alternatively, you can download it manually:

```bash
python scripts/download_dataset.py --path ./data
```

## 🚀 Quick Start

### 1. Data Exploration

Generate and run the data exploration notebook:

```bash
python create_notebook.py
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Train Your First Model (Custom CNN)

```bash
# Train a custom CNN with default settings
python scripts/train_custom_cnn.py

# Train with custom configuration
python scripts/train_custom_cnn.py --config configs/custom_cnn_config.yaml
```

### 3. Train a Pre-trained Model

```bash
# Train ResNet18 with pre-trained weights
python scripts/train_resnet_cifar100.py --model-depth 18 --pretrained

# Train Vision Transformer
python scripts/train_vision_transformer.py --config vit_pretrained
```

## 🏗️ Available Models

### Convolutional Neural Networks
- **Custom CNN**: Simple 4-layer CNN designed for CIFAR-100
- **ResNet**: ResNet-18, ResNet-34, ResNet-50 with CIFAR adaptations
- **VGG**: VGG-16, VGG-19 with modified classifier
- **DenseNet**: DenseNet-121 with efficient feature reuse
- **EfficientNet**: EfficientNet-B0 to B3 with compound scaling

### Modern Architectures
- **ConvNeXt**: Modern CNN inspired by Vision Transformers
- **Vision Transformer (ViT)**: Transformer architecture for images
- **Swin Transformer**: Hierarchical vision transformer

### Classical ML Baseline
- **SVM**: Feature extraction with pre-trained CNN + SVM classifier

## 🎯 Training Models

### Custom CNN

```bash
# Basic training
python scripts/train_custom_cnn.py

# With specific parameters
python scripts/train_custom_cnn.py \
    --batch-size 128 \
    --config configs/custom_cnn_config.yaml

# Create default config file
python scripts/train_custom_cnn.py --create-config
```

### ResNet Models

```bash
# Train ResNet-18
python scripts/train_resnet_cifar100.py --model-depth 18

# Train ResNet-50 without pre-trained weights
python scripts/train_resnet_cifar100.py \
    --model-depth 50 \
    --no-pretrained \
    --config configs/resnet_config.yaml
```

### Vision Transformers

```bash
# Train pre-trained ViT
python scripts/train_vision_transformer.py --config vit_pretrained

# Train custom small ViT
python scripts/train_vision_transformer.py --config vit_small_custom

# Compare multiple configurations
python scripts/train_vision_transformer.py --config all
```

### Swin Transformer

```bash
# Train Swin Transformer Tiny
python scripts/train_swin_transformer.py \
    --model-size tiny \
    --batch-size 64 \
    --pretrained

# Train with custom config
python scripts/train_swin_transformer.py \
    --config configs/swin_transformer_config.yaml
```

### ConvNeXt

```bash
# Train single ConvNeXt model
python scripts/train_convnext.py --variant tiny

# Compare all variants
python scripts/train_convnext.py --compare-all

# Custom training with specific parameters
python scripts/train_convnext.py \
    --variant small \
    --epochs 100 \
    --lr 0.001 \
    --batch-size 64
```

### EfficientNet

```bash
# Train EfficientNet-B0
python scripts/train_efficientnet.py --variant b0

# Compare multiple variants
python scripts/train_efficientnet.py --compare --variants b0 b1 b2

# Train with specific parameters
python scripts/train_efficientnet.py \
    --variant b1 \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001
```

### DenseNet

```bash
# Train DenseNet-121
python scripts/train_densenet.py \
    --batch-size 64 \
    --epochs 50 \
    --pretrained
```

### SVM Baseline

```bash
# Train SVM with ResNet-18 features
python src/models/svm_baseline.py
```

## 📊 Evaluation and Analysis

### Model Analysis

```bash
# Analyze ResNet models
python scripts/analyze_resnet.py \
    --models resnet18_pretrained resnet34_pretrained resnet50_pretrained

# Analyze DenseNet performance
python scripts/analyze_densenet.py \
    --model-path models/densenet121_best.pth

# Compare ResNet variants
python scripts/compare_resnet.py \
    --models resnet18_pretrained resnet34_pretrained
```

### Evaluation Only

```bash
# Evaluate trained model without retraining
python scripts/train_custom_cnn.py \
    --eval-only \
    --model-path models/custom_cnn_best.pth
```

## ⚙️ Configuration

### Configuration Files

Each model has its own YAML configuration file in the `configs/` directory:

- `custom_cnn_config.yaml`: Custom CNN settings
- `resnet_config.yaml`: ResNet training parameters
- `swin_transformer_config.yaml`: Swin Transformer configuration
- `convnext_config.yaml`: ConvNeXt comprehensive settings

### Key Configuration Parameters

```yaml
# Training settings
num_epochs: 50
batch_size: 128
learning_rate: 0.001
weight_decay: 1e-4
optimizer: adamw  # adam, adamw, sgd

# Data augmentation
augmentation:
  random_crop: true
  random_horizontal_flip: true
  color_jitter: true
  mixup: false
  cutmix: false

# Learning rate scheduling
scheduler: reduce_on_plateau  # cosine, step, none
patience: 7  # Early stopping patience
```

### Creating Custom Configurations

```bash
# Generate default config for custom CNN
python scripts/train_custom_cnn.py --create-config

# Copy and modify existing configs
cp configs/resnet_config.yaml configs/my_custom_config.yaml
# Edit the file with your preferred settings
```

## 📈 Results

After training, results are saved in the `results/` directory:

```
results/
├── plots/                     # Training curves and visualizations
├── model_analysis/           # Detailed model analysis
├── confusion_matrices/       # Confusion matrix plots
└── comparison/              # Model comparison results
```

### Key Output Files

- `{model_name}_training_history.png`: Training and validation curves
- `{model_name}_confusion_matrix.png`: Confusion matrix visualization
- `{model_name}_class_metrics.csv`: Per-class performance metrics
- `{model_name}_results.pkl`: Complete results dictionary
- `model_comparison.csv`: Comparison across different models

### Expected Performance

| Model | Top-1 Accuracy | Top-5 Accuracy | Parameters |
|-------|---------------|---------------|------------|
| Custom CNN | ~45-55% | ~75-80% | ~2M |
| ResNet-18 | ~65-70% | ~85-90% | ~11M |
| ResNet-50 | ~70-75% | ~90-92% | ~23M |
| DenseNet-121 | ~70-75% | ~90-92% | ~7M |
| EfficientNet-B0 | ~75-80% | ~92-95% | ~5M |
| Vision Transformer | ~70-75% | ~88-92% | ~86M |
| Swin Transformer | ~75-80% | ~90-93% | ~28M |
| ConvNeXt-Tiny | ~75-80% | ~92-95% | ~28M |

## 🔧 Advanced Usage

### Multi-GPU Training

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train_resnet_cifar100.py

# For multiple GPUs (if implemented)
python -m torch.distributed.launch --nproc_per_node=2 scripts/train_resnet_cifar100.py
```

### Custom Data Augmentation

Modify the configuration files to enable/disable specific augmentations:

```yaml
augmentation:
  mixup: true
  cutmix: true
  autoaugment: true
  random_erase: true
```

### Hyperparameter Tuning

```bash
# Try different learning rates
for lr in 0.001 0.0001 0.01; do
    python scripts/train_custom_cnn.py --lr $lr
done
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Import Errors**: Ensure all dependencies are installed and paths are correct
3. **Slow Training**: Use GPU if available, reduce image size, or use smaller models
4. **Poor Performance**: Try data augmentation, learning rate scheduling, or pre-trained models

### Getting Help

```bash
# Check imports and model creation
python scripts/test_imports.py

# Verify environment setup
python scripts/setup_environment.py --check
```

## 📚 Additional Resources

- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Training! 🚀**