import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
from typing import Optional, Literal, Tuple
from transformers import ViTModel, SwinModel
from .metric_learning import CombinedMetricLoss
import warnings

class LocalFeatureModule(nn.Module):
    """Module for extracting and processing local features"""
    
    def __init__(self, feature_type: str = 'sift', num_features: int = 2000):
        super().__init__()
        self.feature_type = feature_type
        self.num_features = num_features
        
        # Initialize feature extractor
        if feature_type == 'sift':
            self.feature_extractor = cv2.SIFT_create(nfeatures=num_features)
        elif feature_type == 'orb':
            self.feature_extractor = cv2.ORB_create(nfeatures=num_features)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Learnable aggregation layer
        self.feature_dim = 128 if feature_type == 'sift' else 32  # ORB has 32-dim descriptors
        self.aggregator = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and process local features
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Processed local features (B, 512)
        """
        batch_size = x.size(0)
        device = x.device
        features_list = []

        # Process each image in batch
        for img in x.cpu().numpy().transpose(0, 2, 3, 1):
            # Convert to grayscale and uint8
            img_gray = (img * 255).astype(np.uint8)
            if img_gray.shape[-1] == 3:
                img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
            
            # Extract keypoints and descriptors
            _, descriptors = self.feature_extractor.detectAndCompute(img_gray, None)
            
            # Handle case when no features are detected
            if descriptors is None:
                descriptors = np.zeros((1, self.feature_dim), dtype=np.float32)
            
            # Ensure fixed number of features through random sampling or padding
            if len(descriptors) > self.num_features:
                indices = np.random.choice(len(descriptors), self.num_features, replace=False)
                descriptors = descriptors[indices]
            elif len(descriptors) < self.num_features:
                padding = np.zeros((self.num_features - len(descriptors), self.feature_dim))
                descriptors = np.vstack([descriptors, padding])
            
            features_list.append(descriptors)

        # Convert to tensor and process through aggregator
        features = torch.from_numpy(np.stack(features_list)).float().to(device)
        features = features.view(batch_size, self.num_features, -1)
        features = features.mean(dim=1)  # Simple average pooling
        features = self.aggregator(features)
        
        return features

class PawprintModel(nn.Module):
    """Base model for pawprint identification combining CNN and local features"""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        mode: Literal["full", "linear", "metric"] = "full",
        metric_learning_config: Optional[dict] = None,
        use_local_features: bool = False,
        local_feature_type: str = 'sift',
        num_local_features: int = 2000
    ):
        """
        Args:
            num_classes: Number of classes to classify
            backbone: Name of the backbone model
            pretrained: Whether to use pretrained weights
            mode: Training mode
                - "full": Full fine-tuning with CE loss
                - "linear": Linear probing (frozen backbone)
                - "metric": Metric learning with projection head
            local_feature_type: Type of local feature extraction
            metric_learning_config: Configuration for metric learning
            use_local_features: Whether to use local features
            num_local_features: Number of local features
        """
        super().__init__()
        self.mode = mode
        
        # Initialize backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            self.feat_dim = 2048
            if mode == "linear":
                for param in self.backbone.parameters():
                    param.requires_grad = False
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            
        elif backbone == "vgg16":
            self.backbone = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
            self.feat_dim = 4096
            if mode == "linear":
                for param in self.backbone.parameters():
                    param.requires_grad = False
            self.backbone = nn.Sequential(
                self.backbone.features,
                self.backbone.avgpool,
                nn.Flatten(),
                *list(self.backbone.classifier.children())[:-1]
            )
            
        elif backbone == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
            self.feat_dim = 1280
            if mode == "linear":
                for param in self.backbone.parameters():
                    param.requires_grad = False
            self.backbone = nn.Sequential(
                self.backbone.features,
                self.backbone.avgpool,
                nn.Flatten()
            )
            
        elif backbone == "vit_b":
            self.backbone = ViTModel.from_pretrained(
                'google/vit-base-patch16-224', add_pooling_layer=True
            )
            self.feat_dim = 768
            if mode == "linear":
                for param in self.backbone.parameters():
                    param.requires_grad = False
            # Metric learning is supported in 'metric' mode via the common projection head below.
            
        elif backbone == "swin_b":
            self.backbone = SwinModel.from_pretrained(
                'microsoft/swin-base-patch4-window7-224', add_pooling_layer=True
            )
            self.feat_dim = 1024
            if mode == "linear":
                for param in self.backbone.parameters():
                    param.requires_grad = False
            # Metric learning is supported in 'metric' mode via the common projection head below.
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Add local feature module
        self.use_local_features = use_local_features
        if use_local_features:
            self.local_feature_module = LocalFeatureModule(
                feature_type=local_feature_type,
                num_features=num_local_features
            )
            # Update feature dimension to include both backbone and local features
            self.combined_feat_dim = self.feat_dim + 512  # 512 is local feature dim
        else:
            self.combined_feat_dim = self.feat_dim

        # Update classifier for combined features
        self.classifier = nn.Linear(self.combined_feat_dim, num_classes)
        
        # Add metric learning components if in metric mode
        self.use_metric_learning = (mode == "metric")
        if self.use_metric_learning:
            # Projection head for metric learning
            self.projection = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, self.feat_dim)
            )
            
            self.metric_learning = CombinedMetricLoss(
                feat_dim=self.feat_dim,
                num_classes=num_classes,
                **(metric_learning_config or {})
            )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass combining CNN and local features"""
        # Extract CNN features
        cnn_features = self.backbone(x)
        if isinstance(cnn_features, dict):
            cnn_features = cnn_features.pooler_output
        if len(cnn_features.shape) > 2:
            cnn_features = cnn_features.view(cnn_features.size(0), -1)
            
        # Extract and combine local features if enabled
        if self.use_local_features:
            local_features = self.local_feature_module(x)
            features = torch.cat([cnn_features, local_features], dim=1)
        else:
            features = cnn_features
        
        # Get classification logits
        logits = self.classifier(features)
        
        return logits, features
    
    def get_trainable_params(self):
        """Returns parameters that should be trained"""
        if self.mode == "linear":
            return self.classifier.parameters()
        return self.parameters() 