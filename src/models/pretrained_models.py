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
    """Vision Transformer (ViT) model for CIFAR-100 classification (using torchvision)"""
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


# Custom ViT Models (will try to import, fallback to None if not available)
try:
    from .vision_transformer import VisionTransformer, create_vit_model
    
    class CustomVisionTransformerModel(nn.Module):
        """Custom Vision Transformer model for CIFAR-100 classification"""
        def __init__(self, num_classes=100, model_size='tiny', pretrained=False):
            super(CustomVisionTransformerModel, self).__init__()
            
            if pretrained:
                # Use torchvision's pretrained ViT and adapt for CIFAR-100
                self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
                self.model.heads[0] = nn.Linear(768, num_classes)
                self.resize_input = True
            else:
                # Use our custom implementation
                self.model = create_vit_model(model_size, num_classes)
                self.resize_input = False
            
        def forward(self, x):
            if self.resize_input:
                # Resize input for pretrained models that expect 224x224
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            return self.model(x)
    
    CUSTOM_VIT_AVAILABLE = True
    
except ImportError:
    print("Custom ViT implementation not available. Only torchvision ViT will be used.")
    CustomVisionTransformerModel = None
    CUSTOM_VIT_AVAILABLE = False


def get_model(model_name, num_classes=100, pretrained=True, **kwargs):
    """
    Factory function to get a specified model
    
    Args:
        model_name: Name of the model to instantiate
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments (e.g., model_size for ViT)
        
    Returns:
        An instantiated model
    """
    # Import custom CNN here to avoid circular imports
    try:
        from .custom_cnn import CustomCNN
    except ImportError:
        print("Warning: CustomCNN not available")
        CustomCNN = None
    
    # Get model size for ViT models
    model_size = kwargs.get('model_size', 'tiny')
    
    models_dict = {
        'resnet18': lambda: ResNet18Model(num_classes=num_classes, pretrained=pretrained),
        'resnet50': lambda: ResNet50Model(num_classes=num_classes, pretrained=pretrained),
        'vgg16': lambda: VGG16Model(num_classes=num_classes, pretrained=pretrained),
        'densenet121': lambda: DenseNet121Model(num_classes=num_classes, pretrained=pretrained),
        'efficientnet_b0': lambda: EfficientNetB0Model(num_classes=num_classes, pretrained=pretrained),
        'convnext': lambda: ConvNextModel(num_classes=num_classes, pretrained=pretrained),
        'vit': lambda: VisionTransformerModel(num_classes=num_classes, pretrained=pretrained),
        'swin': lambda: SwinTransformerModel(num_classes=num_classes, pretrained=pretrained),
    }
    
    # Add custom CNN if available
    if CustomCNN is not None:
        models_dict['custom_cnn'] = lambda: CustomCNN(num_classes=num_classes)
    
    # Add custom ViT models if available
    if CUSTOM_VIT_AVAILABLE and CustomVisionTransformerModel is not None:
        models_dict.update({
            'vit_custom': lambda: CustomVisionTransformerModel(num_classes=num_classes, model_size=model_size, pretrained=pretrained),
            'vit_tiny': lambda: CustomVisionTransformerModel(num_classes=num_classes, model_size='tiny', pretrained=False),
            'vit_small': lambda: CustomVisionTransformerModel(num_classes=num_classes, model_size='small', pretrained=False),
            'vit_base': lambda: CustomVisionTransformerModel(num_classes=num_classes, model_size='base', pretrained=False),
        })
    
    if model_name not in models_dict:
        available_models = list(models_dict.keys())
        raise ValueError(f"Model '{model_name}' not implemented. Available models: {available_models}")
    
    return models_dict[model_name]()


def list_available_models():
    """
    List all available models
    
    Returns:
        List of available model names
    """
    base_models = ['resnet18', 'resnet50', 'vgg16', 'densenet121', 'efficientnet_b0', 'convnext', 'vit', 'swin']
    
    # Add custom CNN if available
    try:
        from .custom_cnn import CustomCNN
        base_models.append('custom_cnn')
    except ImportError:
        pass
    
    # Add custom ViT models if available
    if CUSTOM_VIT_AVAILABLE:
        base_models.extend(['vit_custom', 'vit_tiny', 'vit_small', 'vit_base'])
    
    return base_models


# Example usage
if __name__ == "__main__":
    import time
    
    # Test each model with a small batch of data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get available models
    available_models = list_available_models()
    print(f"Available models: {available_models}")
    
    # Test models
    for model_name in available_models[:5]:  # Test first 5 models
        print(f"\nTesting model: {model_name}")
        
        try:
            # Create model
            if 'vit_custom' in model_name or 'vit_tiny' in model_name or 'vit_small' in model_name or 'vit_base' in model_name:
                model = get_model(model_name, model_size='tiny')
            else:
                model = get_model(model_name)
            
            # Move model to device
            model = model.to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Test with dummy data
            dummy_input = torch.randn(4, 3, 32, 32).to(device)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                outputs = model(dummy_input)
            end_time = time.time()
            
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
    
    print("\nModel testing completed!")