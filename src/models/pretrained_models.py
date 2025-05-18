# CIFAR-100 Image Classification Project - Pre-trained Models Implementation

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class ResNet18Model(nn.Module):
    """ResNet18 model for CIFAR-100 classification"""
    def __init__(self, num_classes=100, pretrained=True):
        super(ResNet18Model, self).__init__()
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                self.model = models.resnet18(weights=weights)
            except (AttributeError, ImportError):
                # Fallback for older torch versions
                self.model = models.resnet18(pretrained=True)
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
            try:
                weights = models.ResNet50_Weights.DEFAULT
                self.model = models.resnet50(weights=weights)
            except (AttributeError, ImportError):
                # Fallback for older torch versions
                self.model = models.resnet50(pretrained=True)
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
            try:
                weights = models.VGG16_Weights.DEFAULT
                self.model = models.vgg16(weights=weights)
            except (AttributeError, ImportError):
                # Fallback for older torch versions
                self.model = models.vgg16(pretrained=True)
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
            try:
                weights = models.DenseNet121_Weights.DEFAULT
                self.model = models.densenet121(weights=weights)
            except (AttributeError, ImportError):
                # Fallback for older torch versions
                self.model = models.densenet121(pretrained=True)
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
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.model = models.efficientnet_b0(weights=weights)
            except (AttributeError, ImportError):
                # Fallback for older torch versions
                self.model = models.efficientnet_b0(pretrained=True)
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
            try:
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT
                self.model = models.convnext_tiny(weights=weights)
            except (AttributeError, ImportError):
                # Fallback for older torch versions
                try:
                    self.model = models.convnext_tiny(pretrained=True)
                except:
                    print("ConvNeXt not available in this PyTorch version, using without pre-training")
                    self.model = models.convnext_tiny(pretrained=False)
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
            try:
                weights = models.ViT_B_16_Weights.DEFAULT
                self.model = models.vit_b_16(weights=weights)
            except (AttributeError, ImportError):
                # Fallback for older torch versions
                try:
                    self.model = models.vit_b_16(pretrained=True)
                except:
                    print("ViT not available in this PyTorch version, using without pre-training")
                    self.model = models.vit_b_16(pretrained=False)
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
            try:
                weights = models.Swin_T_Weights.DEFAULT
                self.model = models.swin_t(weights=weights)
            except (AttributeError, ImportError):
                # Fallback for older torch versions
                try:
                    self.model = models.swin_t(pretrained=True)
                except:
                    print("Swin Transformer not available in this PyTorch version, using without pre-training")
                    self.model = models.swin_t(pretrained=False)
        else:
            self.model = models.swin_t(weights=None)
            
        # Replace head
        self.model.head = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # Swin Transformer expects 224x224 images, so we need to resize
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)