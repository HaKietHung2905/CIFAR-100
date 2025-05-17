# CIFAR-100 Image Classification Project - Pre-trained Models Implementation

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, ResNet50_Weights, VGG16_Weights
from torchvision.models import DenseNet121_Weights, EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights

class ResNet18Model(nn.Module):
    """ResNet18 model for CIFAR-100 classification"""
    def __init__(self, num_classes=100, pretrained=True):
        super(ResNet18Model, self).__init__()
        if pretrained:
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=None)
            
        # Replace the first conv layer to handle 32x32 input size
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool layer for CIFAR
        
        # Replace the final fc layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)


class ResNet50Model(nn.Module):
    """ResNet50 model for CIFAR-100 classification"""
    def __init__(self, num_classes=100, pretrained=True):
        super(ResNet50Model, self).__init__()
        if pretrained:
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50(weights=None)
            
        # Replace the first conv layer to handle 32x32 input size
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool layer for CIFAR
        
        # Replace the final fc layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)


class VGG16Model(nn.Module):
    """VGG16 model for CIFAR-100 classification"""
    def __init__(self, num_classes=100, pretrained=True):
        super(VGG16Model, self).__init__()
        if pretrained:
            self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        else:
            self.model = models.vgg16(weights=None)
            
        # Replace classifier for num_classes
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        return self.model(x)


class DenseNet121Model(nn.Module):
    """DenseNet121 model for CIFAR-100 classification"""
    def __init__(self, num_classes=100, pretrained=True):
        super(DenseNet121Model, self).__init__()
        if pretrained:
            self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            self.model = models.densenet121(weights=None)
            
        # Replace the classifier
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)


class EfficientNetB0Model(nn.Module):
    """EfficientNetB0 model for CIFAR-100 classification"""
    def __init__(self, num_classes=100, pretrained=True):
        super(EfficientNetB0Model, self).__init__()
        if pretrained:
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            self.model = models.efficientnet_b0(weights=None)
            
        # Replace classifier head
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)


class ConvNextModel(nn.Module):
    """ConvNeXt model for CIFAR-100 classification"""
    def __init__(self, num_classes=100, pretrained=True):
        super(ConvNextModel, self).__init__()
        if pretrained:
            self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        else:
            self.model = models.convnext_tiny(weights=None)
            
        # Replace classifier head
        self.model.classifier[2] = nn.Linear(768, num_classes)
        
    def forward(self, x):
        return self.model(x)


class VisionTransformerModel(nn.Module):
    """Vision Transformer (ViT) model for CIFAR-100 classification"""
    def __init__(self, num_classes=100, pretrained=True):
        super(VisionTransformerModel, self).__init__()
        if pretrained:
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        else:
            self.model = models.vit_b_16(weights=None)
            
        # Replace classifier head
        self.model.heads[0] = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # ViT expects 224x224 images, so we need to resize
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)


class SwinTransformerModel(nn.Module):
    """Swin Transformer model for CIFAR-100 classification"""
    def __init__(self, num_classes=100, pretrained=True):
        super(SwinTransformerModel, self).__init__()
        if pretrained:
            self.model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        else:
            self.model = models.swin_t(weights=None)
            
        # Replace head
        self.model.head = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # Swin Transformer expects 224x224 images, so we need to resize
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)


def get_model(model_name, num_classes=100, pretrained=True):
    """
    Factory function to get a specified model
    
    Args:
        model_name: Name of the model to instantiate
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        An instantiated model
    """
    models_dict = {
        'custom_cnn': lambda: CustomCNN(num_classes=num_classes),
        'resnet18': lambda: ResNet18Model(num_classes=num_classes, pretrained=pretrained),
        'resnet50': lambda: ResNet50Model(num_classes=num_classes, pretrained=pretrained),
        'vgg16': lambda: VGG16Model(num_classes=num_classes, pretrained=pretrained),
        'densenet121': lambda: DenseNet121Model(num_classes=num_classes, pretrained=pretrained),
        'efficientnet_b0': lambda: EfficientNetB0Model(num_classes=num_classes, pretrained=pretrained),
        'convnext': lambda: ConvNextModel(num_classes=num_classes, pretrained=pretrained),
        'vit': lambda: VisionTransformerModel(num_classes=num_classes, pretrained=pretrained),
        'swin': lambda: SwinTransformerModel(num_classes=num_classes, pretrained=pretrained),
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model '{model_name}' not implemented. Available models: {list(models_dict.keys())}")
    
    return models_dict[model_name]()


# Example usage
if __name__ == "__main__":
    from data_preparation import load_cifar100
    import time
    
    train_loader, test_loader, classes = load_cifar100(batch_size=4)
    
    # Test each model with a small batch of data
    for model_name in ['custom_cnn', 'resnet18', 'vgg16', 'densenet121', 
                      'efficientnet_b0', 'convnext', 'vit', 'swin']:
        print(f"\nTesting model: {model_name}")
        model = get_model(model_name)
        
        # Move model to GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Get a batch of data
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        
        # Measure inference time
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()
        
        print(f"Input shape: {images.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")