import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class TripletLoss(nn.Module):
    """
    Triplet loss with hard mining
    margin: Minimum distance between positive and negative pairs
    mining_strategy: 'batch_hard' or 'batch_all'
    """
    def __init__(self, margin: float = 0.3, mining_strategy: str = 'batch_hard'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Compute pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings)
        
        # Get positive and negative masks
        labels_mat = labels.view(-1, 1)
        positive_mask = labels_mat == labels_mat.t()  # shape: (B, B)
        negative_mask = ~positive_mask               # shape: (B, B)
        
        if self.mining_strategy == 'batch_hard':
            # Hardest positive for each anchor
            # (if you want to exclude the anchor, you can handle diagonal=0 or adjust the mask)
            hardest_positive_dist = (
                pairwise_dist * positive_mask.float()
            ).max(dim=1)[0]
            
            # Hardest negative for each anchor
            hardest_negative_dist = (
                pairwise_dist + (1e6 * positive_mask.float())
            ).min(dim=1)[0]
            
            # Compute triplet loss
            losses = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            loss = losses.mean()
            
            # Calculate metrics
            metrics = {
                'avg_positive_dist': hardest_positive_dist.mean().item(),
                'avg_negative_dist': hardest_negative_dist.mean().item(),
                'triplet_loss': loss.item()
            }
            
        else:  # batch_all
            # Compute valid triplets mask
            valid_triplets = (
                positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)
            )
            
            # Compute triplet loss for all valid triplets
            triplet_loss = (
                pairwise_dist.unsqueeze(2) - pairwise_dist.unsqueeze(1) + self.margin
            )
            
            # Apply masking and compute final loss
            triplet_loss = F.relu(triplet_loss) * valid_triplets.float()
            num_triplets = valid_triplets.sum()
            
            if num_triplets == 0:
                loss = torch.tensor(0.0, device=embeddings.device)
            else:
                loss = triplet_loss.sum() / num_triplets
                
            metrics = {
                'num_valid_triplets': num_triplets.item(),
                'triplet_loss': loss.item()
            }
            
        return loss, metrics

class CenterLoss(nn.Module):
    """
    Center loss for learning class centers
    alpha: Learning rate for centers
    """
    def __init__(self, feat_dim: int, num_classes: int, alpha: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Initialize centers
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim)
        )
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Get centers for current batch
        batch_centers = self.centers[labels]
        
        # Compute center loss
        center_loss = (features - batch_centers).pow(2).sum(dim=1).mean()
        
        # Update centers (during training only)
        if self.training:
            # Compute class appearance mask
            classes_batch = torch.zeros(
                self.num_classes, device=features.device
            ).scatter_(0, labels, 1)
            
            # Compute class centers delta
            centers_delta = torch.zeros_like(self.centers)
            for i in range(self.num_classes):
                if classes_batch[i] > 0:
                    class_mask = (labels == i)
                    class_features = features[class_mask]
                    centers_delta[i] = (
                        class_features.mean(dim=0) - self.centers[i]
                    ) * self.alpha
                    
            self.centers.data.add_(centers_delta)
            
        metrics = {'center_loss': center_loss.item()}
        return center_loss, metrics



class ArcFaceLoss(nn.Module):
    """
    ArcFace loss with cosine margin (numerically stable version)
    """
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # 1) Normalize features & weights
        features_norm = F.normalize(features, p=2, dim=1)    # (B, feat_dim)
        weights_norm = F.normalize(self.weight, p=2, dim=1)  # (num_classes, feat_dim)
        
        # 2) Compute cosine similarity
        cos_theta = F.linear(features_norm, weights_norm)    # (B, num_classes)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
        
        # 3) Apply arc margin only to the ground-truth class
        #    cos(theta + margin) = cos(theta)*cos(margin) - sin(theta)*sin(margin)
        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta + 1e-7)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # 4) Replace only the logits of the correct class with the margin version
        logits = cos_theta.clone()
        batch_indices = torch.arange(labels.size(0), device=labels.device)
        logits[batch_indices, labels] = cos_theta_m[batch_indices, labels]
        
        # 5) Scale & compute cross-entropy
        logits *= self.scale
        loss = F.cross_entropy(logits, labels)
        
        metrics = {
            'arcface_loss': loss.item(),
            'avg_cos_sim': cos_theta.mean().item()
        }
        return loss, metrics

class CombinedMetricLoss(nn.Module):
    """
    Combines multiple metric learning losses
    """
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        triplet_weight: float = 1.0,
        center_weight: float = 0.1,
        arcface_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.triplet_weight = triplet_weight
        self.center_weight = center_weight
        self.arcface_weight = arcface_weight
        
        # Initialize individual losses
        self.triplet_loss = TripletLoss(**kwargs.get('triplet', {}))
        self.center_loss = CenterLoss(feat_dim, num_classes, **kwargs.get('center', {}))
        self.arcface_loss = ArcFaceLoss(feat_dim, num_classes, **kwargs.get('arcface', {}))
        
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        # Compute individual losses
        triplet_loss, triplet_metrics = self.triplet_loss(features, labels)
        center_loss, center_metrics = self.center_loss(features, labels)
        arcface_loss, arcface_metrics = self.arcface_loss(features, labels)
        
        # Combine losses
        total_loss = (
            self.triplet_weight * triplet_loss +
            self.center_weight * center_loss +
            self.arcface_weight * arcface_loss
        )
        
        # Combine metrics
        metrics = {
            'total_loss': total_loss.item(),
            **{f'triplet_{k}': v for k, v in triplet_metrics.items()},
            **{f'center_{k}': v for k, v in center_metrics.items()},
            **{f'arcface_{k}': v for k, v in arcface_metrics.items()}
        }
        
        return total_loss, metrics
    

if __name__ == "__main__":
    # Example hyperparameters
    batch_size = 8
    feat_dim = 128
    num_classes = 4
    
    # Prepare random data
    features = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, size=(batch_size,))
    
    # Initialize CombinedMetricLoss
    combined_loss_fn = CombinedMetricLoss(
        feat_dim=feat_dim,
        num_classes=num_classes,
        triplet_weight=1.0,
        center_weight=0.1,
        arcface_weight=0.1,
        triplet={'margin': 0.3, 'mining_strategy': 'batch_hard'},
        center={'alpha': 0.1},
        arcface={'scale': 30.0, 'margin': 0.5}
    )
    
    # Calculate loss
    total_loss, metrics = combined_loss_fn(features, labels)
    
    print("Total loss:", total_loss.item())
    print("Metrics:", metrics)