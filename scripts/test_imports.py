#!/usr/bin/env python
# Test script to check imports

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing the model classes
try:
    from src.models.pretrained_models import ResNet18Model, ResNet50Model
    print("✅ Successfully imported ResNet18Model and ResNet50Model")
except ImportError as e:
    print(f"❌ Error importing from pretrained_models: {e}")

# Try importing ResNet34
try:
    from src.models.resnet34 import create_resnet34
    print("✅ Successfully imported create_resnet34")
except ImportError as e:
    print(f"❌ Error importing from resnet34: {e}")

# Try importing data preparation
try:
    from src.data.data_preparation import load_cifar100
    print("✅ Successfully imported load_cifar100")
except ImportError as e:
    print(f"❌ Error importing from data_preparation: {e}")

# Try creating the models
try:
    import torch
    from src.models.pretrained_models import ResNet18Model
    from src.models.resnet34 import create_resnet34  
    from src.models.pretrained_models import ResNet50Model
    
    model18 = ResNet18Model(num_classes=100)
    model34 = create_resnet34(num_classes=100)
    model50 = ResNet50Model(num_classes=100)
    print(f"✅ Successfully created all models")
    
    print(f"ResNet18 parameters: {sum(p.numel() for p in model18.parameters()):,}")
    print(f"ResNet34 parameters: {sum(p.numel() for p in model34.parameters()):,}")
    print(f"ResNet50 parameters: {sum(p.numel() for p in model50.parameters()):,}")
    
except Exception as e:
    print(f"❌ Error creating models: {e}")

print("\nImport test completed. If you see all ✅, the circular import issue is fixed!")