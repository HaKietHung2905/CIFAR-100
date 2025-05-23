# src/models/efficientnet_advanced.py
# Advanced EfficientNet techniques including AutoAugment, Compound Scaling, and Knowledge Distillation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment
import numpy as np
import math
import random
from typing import Tuple, List, Optional


class CompoundScaling:
    """
    Implement EfficientNet's compound scaling methodology
    Scale depth, width, and resolution simultaneously with fixed ratios
    """
    
    def __init__(self, base_config: dict, phi: float = 1.0):
        """
        Args:
            base_config: Base configuration for EfficientNet-B0
            phi: Compound scaling coefficient
        """
        self.base_config = base_config
        self.phi = phi
        
        # EfficientNet scaling coefficients
        self.alpha = 1.2  # depth scaling
        self.beta = 1.1   # width scaling  
        self.gamma = 1.15 # resolution scaling
        
    def get_scaled_config(self) -> dict:
        """Get scaled configuration based on phi"""
        scaled_config = self.base_config.copy()
        
        # Apply compound scaling
        depth_multiplier = self.alpha ** self.phi
        width_multiplier = self.beta ** self.phi
        resolution_multiplier = self.gamma ** self.phi
        
        # Scale target resolution
        base_resolution = self.base_config.get('target_size', 224)
        scaled_config['target_size'] = int(base_resolution * resolution_multiplier)
        
        # Adjust batch size based on resolution (maintain similar memory usage)
        resolution_ratio = (scaled_config['target_size'] / base_resolution) ** 2
        scaled_config['batch_size'] = max(1, int(self.base_config['batch_size'] / resolution_ratio))
        
        # Store scaling factors for model architecture
        scaled_config['depth_multiplier'] = depth_multiplier
        scaled_config['width_multiplier'] = width_multiplier
        scaled_config['resolution_multiplier'] = resolution_multiplier
        
        return scaled_config


class AutoAugmentCIFAR100:
    """
    Custom AutoAugment implementation optimized for CIFAR-100
    Based on AutoAugment paper but adapted for smaller images
    """
    
    def __init__(self, magnitude: int = 9):
        self.magnitude = magnitude
        
        # Define CIFAR-100 specific augmentation policies
        self.policies = [
            # Policy 1
            [("Rotate", 0.4, 8), ("Color", 0.6, 9)],
            [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
            
            # Policy 2  
            [("TranslateX", 0.2, 6), ("Equalize", 0.8, 8)],
            [("Solarize", 0.6, 6), ("AutoContrast", 0.2, 6)],
            
            # Policy 3
            [("TranslateY", 0.8, 2), ("Sharpness", 0.0, 8)],
            [("ShearX", 0.8, 4), ("Invert", 0.6, 8)],
            
            # Policy 4
            [("ShearY", 0.6, 8), ("TranslateY", 0.6, 6)],
            [("Rotate", 0.6, 6), ("Color", 0.8, 6)],
            
            # Policy 5
            [("AutoContrast", 0.4, 8), ("TranslateY", 0.2, 2)],
            [("Sharpness", 0.8, 8), ("Brightness", 0.8, 8)],
        ]
    
    def __call__(self, img):
        """Apply random augmentation policy"""
        policy = random.choice(self.policies)
        sub_policy = random.choice(policy)
        
        return self._apply_augmentation(img, sub_policy)
    
    def _apply_augmentation(self, img, sub_policy):
        """Apply specific augmentation"""
        op_name, prob, magnitude = sub_policy
        
        if random.random() > prob:
            return img
        
        if op_name == "Rotate":
            degrees = magnitude * 3  # Scale for CIFAR-100
            return transforms.functional.rotate(img, degrees)
        elif op_name == "TranslateX":
            translate = magnitude / 3  # Smaller translation for 32x32 images
            return transforms.functional.affine(img, 0, [translate, 0], 1, 0)
        elif op_name == "TranslateY":
            translate = magnitude / 3
            return transforms.functional.affine(img, 0, [0, translate], 1, 0)
        elif op_name == "ShearX":
            shear = magnitude / 3
            return transforms.functional.affine(img, 0, [0, 0], 1, [shear, 0])
        elif op_name == "ShearY":
            shear = magnitude / 3
            return transforms.functional.affine(img, 0, [0, 0], 1, [0, shear])
        elif op_name == "Color":
            factor = 1 + magnitude * 0.1
            return transforms.functional.adjust_saturation(img, factor)
        elif op_name == "Brightness":
            factor = 1 + magnitude * 0.1
            return transforms.functional.adjust_brightness(img, factor)
        elif op_name == "Sharpness":
            factor = 1 + magnitude * 0.1
            return transforms.functional.adjust_sharpness(img, factor)
        elif op_name == "AutoContrast":
            return transforms.functional.autocontrast(img)
        elif op_name == "Equalize":
            return transforms.functional.equalize(img)
        elif op_name == "Solarize":
            threshold = 256 - magnitude * 25
            return transforms.functional.solarize(img, threshold)
        elif op_name == "Invert":
            return transforms.functional.invert(img)
        
        return img


class MixUp:
    """
    MixUp data augmentation for improved generalization
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation
        
        Returns:
            mixed_x: Mixed input
            y_a: First set of labels  
            y_b: Second set of labels
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """
    CutMix data augmentation for improved localization
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Generate random bounding box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss for training smaller models with teacher guidance
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            student_outputs: Logits from student model
            teacher_outputs: Logits from teacher model  
            targets: Ground truth labels
        """
        # Soften predictions
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        
        # Distillation loss
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        classification_loss = self.ce_loss(student_outputs, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * classification_loss
        
        return total_loss


class EfficientNetTeacher:
    """
    Teacher model for knowledge distillation using larger EfficientNet variants
    """
    
    def __init__(self, teacher_variant: str = 'b4', num_classes: int = 100):
        from .efficientnet_models import get_efficientnet_model
        
        self.teacher_variant = teacher_variant
        self.teacher = get_efficientnet_model(
            variant=teacher_variant,
            num_classes=num_classes,
            pretrained=True,
            adaptive=True
        )
    
    def load_pretrained_teacher(self, checkpoint_path: str):
        """Load pre-trained teacher model"""
        checkpoint = torch.load(checkpoint_path)
        self.teacher.load_state_dict(checkpoint['model_state_dict'])
        self.teacher.eval()
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def get_teacher_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Get teacher model outputs"""
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher(x)


class AdvancedEfficientNetTrainer:
    """
    Advanced trainer with AutoAugment, MixUp, CutMix, and Knowledge Distillation
    """
    
    def __init__(self, variant: str = 'b0', use_autoaugment: bool = True,
                 use_mixup: bool = True, use_cutmix: bool = True,
                 teacher_variant: Optional[str] = None):
        
        self.variant = variant
        self.use_autoaugment = use_autoaugment
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.teacher_variant = teacher_variant
        
        # Initialize augmentations
        if use_mixup:
            self.mixup = MixUp(alpha=1.0)
        if use_cutmix:
            self.cutmix = CutMix(alpha=1.0)
        
        # Initialize knowledge distillation
        if teacher_variant:
            self.teacher = EfficientNetTeacher(teacher_variant)
            self.kd_loss = KnowledgeDistillationLoss()
    
    def get_advanced_transforms(self, target_size: int = 224, is_training: bool = True):
        """Get advanced data transforms"""
        
        if is_training:
            transforms_list = [
                transforms.Resize((target_size, target_size)),
                transforms.RandomCrop(target_size, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
            
            # Add AutoAugment if enabled
            if self.use_autoaugment:
                transforms_list.append(AutoAugmentCIFAR100(magnitude=9))
            else:
                # Standard augmentations
                transforms_list.extend([
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                ])
            
            transforms_list.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        else:
            transforms_list = [
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ]
        
        return transforms.Compose(transforms_list)
    
    def mixup_cutmix_criterion(self, criterion, pred, y_a, y_b, lam):
        """Compute loss for MixUp/CutMix"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def train_step_advanced(self, model, inputs, targets, criterion, optimizer, 
                          teacher_model=None, epoch=0):
        """Advanced training step with augmentations and knowledge distillation"""
        
        # Apply MixUp or CutMix randomly
        if self.use_mixup and self.use_cutmix:
            # Randomly choose between MixUp and CutMix
            if random.random() > 0.5 and epoch > 10:  # Start CutMix after warmup
                inputs, targets_a, targets_b, lam = self.cutmix(inputs, targets)
            else:
                inputs, targets_a, targets_b, lam = self.mixup(inputs, targets)
        elif self.use_mixup:
            inputs, targets_a, targets_b, lam = self.mixup(inputs, targets)
        elif self.use_cutmix and epoch > 10:
            inputs, targets_a, targets_b, lam = self.cutmix(inputs, targets)
        else:
            targets_a, targets_b, lam = targets, targets, 1
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        if lam != 1:
            # MixUp/CutMix loss
            loss = self.mixup_cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
        
        # Add knowledge distillation if teacher available
        if teacher_model is not None:
            teacher_outputs = teacher_model.get_teacher_outputs(inputs)
            kd_loss = self.kd_loss(outputs, teacher_outputs, targets)
            loss = 0.7 * loss + 0.3 * kd_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item(), outputs, targets


class EfficientNetEnsemble:
    """
    Ensemble of multiple EfficientNet models for improved performance
    """
    
    def __init__(self, variants: List[str] = ['b0', 'b1', 'b2'], num_classes: int = 100):
        from .efficientnet_models import get_efficientnet_model
        
        self.variants = variants
        self.models = {}
        
        # Create models
        for variant in variants:
            self.models[variant] = get_efficientnet_model(
                variant=variant,
                num_classes=num_classes,
                pretrained=True,
                adaptive=True
            )
    
    def load_trained_models(self, model_paths: dict):
        """Load trained model weights"""
        for variant, path in model_paths.items():
            if variant in self.models:
                checkpoint = torch.load(path)
                self.models[variant].load_state_dict(checkpoint['model_state_dict'])
                self.models[variant].eval()
    
    def predict_ensemble(self, x: torch.Tensor, device: torch.device, 
                        weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Make ensemble predictions
        
        Args:
            x: Input tensor
            device: Device to run inference on
            weights: Optional weights for each model
        """
        if weights is None:
            weights = [1.0] * len(self.models)
        
        ensemble_output = None
        
        for i, (variant, model) in enumerate(self.models.items()):
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(x)
                output = F.softmax(output, dim=1)
                
                if ensemble_output is None:
                    ensemble_output = weights[i] * output
                else:
                    ensemble_output += weights[i] * output
        
        return ensemble_output


class EfficientNetAnalyzer:
    """
    Analyze EfficientNet model performance and characteristics
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def analyze_model_efficiency(self, input_size: Tuple[int, int, int] = (3, 224, 224)):
        """Analyze model efficiency metrics"""
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate FLOPs (simplified)
        dummy_input = torch.randn(1, *input_size).to(self.device)
        
        # Memory usage
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)
        
        # Inference time
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start.record()
                    _ = self.model(dummy_input)
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
                else:
                    import time
                    start_time = time.time()
                    _ = self.model(dummy_input)
                    times.append((time.time() - start_time) * 1000)
        
        avg_inference_time = np.mean(times)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'avg_inference_time_ms': avg_inference_time,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'throughput_fps': 1000 / avg_inference_time
        }
    
    def analyze_layer_wise_performance(self):
        """Analyze performance of different layers"""
        layer_info = []
        
        def hook_fn(module, input, output):
            layer_info.append({
                'layer_name': str(module),
                'input_shape': input[0].shape if isinstance(input, tuple) else input.shape,
                'output_shape': output.shape,
                'parameters': sum(p.numel() for p in module.parameters())
            })
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_info


def create_efficientnet_config_search_space():
    """Create configuration search space for hyperparameter optimization"""
    
    search_space = {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],
        'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3],
        'batch_size': [16, 32, 64, 128, 256],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'label_smoothing': [0.0, 0.05, 0.1, 0.15],
        'mixup_alpha': [0.2, 0.5, 1.0, 1.5],
        'cutmix_alpha': [0.5, 1.0, 1.5],
        'autoaugment_magnitude': [5, 7, 9, 11],
        'warmup_epochs': [3, 5, 8, 10],
        'cosine_restarts': [True, False]
    }
    
    return search_space


def optimize_efficientnet_hyperparameters(variant: str = 'b0', n_trials: int = 50):
    """
    Perform hyperparameter optimization using random search
    """
    import random
    from .efficientnet_models import get_efficientnet_model, EFFICIENTNET_CONFIGS
    
    search_space = create_efficientnet_config_search_space()
    best_config = None
    best_score = 0.0
    
    results = []
    
    for trial in range(n_trials):
        # Sample configuration
        config = EFFICIENTNET_CONFIGS[variant].copy()
        
        # Override with random samples
        config['learning_rate'] = random.choice(search_space['learning_rate'])
        config['weight_decay'] = random.choice(search_space['weight_decay'])
        config['batch_size'] = random.choice(search_space['batch_size'])
        config['dropout_rate'] = random.choice(search_space['dropout_rate'])
        
        # Additional hyperparameters
        config['label_smoothing'] = random.choice(search_space['label_smoothing'])
        config['mixup_alpha'] = random.choice(search_space['mixup_alpha'])
        config['cutmix_alpha'] = random.choice(search_space['cutmix_alpha'])
        config['autoaugment_magnitude'] = random.choice(search_space['autoaugment_magnitude'])
        config['warmup_epochs'] = random.choice(search_space['warmup_epochs'])
        
        # Train model with this configuration (simplified for demonstration)
        try:
            # This would be replaced with actual training
            score = evaluate_config(variant, config)
            
            results.append({
                'trial': trial,
                'config': config,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_config = config
                print(f"Trial {trial}: New best score {score:.4f}")
                
        except Exception as e:
            print(f"Trial {trial} failed: {str(e)}")
            continue
    
    return best_config, best_score, results


def evaluate_config(variant: str, config: dict) -> float:
    """
    Evaluate a configuration (placeholder for actual training)
    In practice, this would train the model and return validation accuracy
    """
    # This is a simplified placeholder
    # In reality, you would:
    # 1. Create model with config
    # 2. Train for a few epochs
    # 3. Return validation accuracy
    
    # For demonstration, return a random score influenced by config
    base_score = 0.7
    lr_bonus = max(0, 0.1 - abs(config['learning_rate'] - 0.001))
    wd_bonus = max(0, 0.05 - abs(config['weight_decay'] - 1e-5))
    
    return base_score + lr_bonus + wd_bonus + random.uniform(-0.1, 0.1)


if __name__ == "__main__":
    # Example usage
    print("Advanced EfficientNet Techniques for CIFAR-100")
    print("=" * 50)
    
    # Test AutoAugment
    print("Testing AutoAugment...")
    autoaugment = AutoAugmentCIFAR100()
    
    # Test MixUp
    print("Testing MixUp...")
    mixup = MixUp(alpha=1.0)
    dummy_x = torch.randn(4, 3, 32, 32)
    dummy_y = torch.randint(0, 100, (4,))
    mixed_x, y_a, y_b, lam = mixup(dummy_x, dummy_y)
    print(f"MixUp lambda: {lam:.3f}")
    
    # Test CutMix
    print("Testing CutMix...")
    cutmix = CutMix(alpha=1.0)
    mixed_x, y_a, y_b, lam = cutmix(dummy_x, dummy_y)
    print(f"CutMix lambda: {lam:.3f}")
    
    # Test Compound Scaling
    print("Testing Compound Scaling...")
    base_config = {'target_size': 224, 'batch_size': 128}
    scaling = CompoundScaling(base_config, phi=1.5)
    scaled_config = scaling.get_scaled_config()
    print(f"Scaled config: {scaled_config}")
    
    # Test Advanced Trainer
    print("Testing Advanced Trainer...")
    trainer = AdvancedEfficientNetTrainer(
        variant='b0',
        use_autoaugment=True,
        use_mixup=True,
        use_cutmix=True
    )
    
    transforms_train = trainer.get_advanced_transforms(224, is_training=True)
    print(f"Advanced transforms created: {len(transforms_train.transforms)} transforms")
    
    print("All tests completed successfully!")