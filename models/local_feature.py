import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Optional, Tuple, List
from sklearn.cluster import MiniBatchKMeans
import warnings

class LocalFeatureModel(nn.Module):
    """Model for pawprint identification using local features"""
    
    def __init__(
        self,
        num_classes: int,
        feature_type: str = 'sift',
        num_features: int = 2000,
        encoding: str = 'learnable',  # ['learnable', 'bovw']
        vocab_size: int = 1000,
        spatial_pyramid: bool = True,
        pyramid_levels: List[int] = [2, 4]
    ):
        """
        Args:
            num_classes: Number of classes to classify
            feature_type: Type of local feature detector ('sift' or 'orb')
            num_features: Maximum number of features to extract per image
            encoding: Feature encoding method
            vocab_size: Vocabulary size for BoVW encoding
            spatial_pyramid: Whether to use spatial pyramid pooling
            pyramid_levels: Levels for spatial pyramid (e.g., [2, 4] means 1x1, 2x2, 4x4)
        """
        super().__init__()
        self.feature_type = feature_type
        self.num_features = num_features
        self.encoding = encoding
        self.spatial_pyramid = spatial_pyramid
        self.pyramid_levels = [1] + pyramid_levels if spatial_pyramid else [1]
        
        # Initialize feature extractor
        if feature_type == 'sift':
            self.feature_extractor = cv2.SIFT_create(nfeatures=num_features)
            self.feature_dim = 128
        elif feature_type == 'orb':
            self.feature_extractor = cv2.ORB_create(nfeatures=num_features)
            self.feature_dim = 32
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Calculate final feature dimension based on encoding method
        if encoding == 'learnable':
            self.final_feat_dim = 512  # Fixed dimension for learnable encoding
        else:  # bovw
            # Calculate total number of spatial bins
            total_spatial_bins = sum(level * level for level in self.pyramid_levels)
            self.final_feat_dim = vocab_size * total_spatial_bins
        
        print(f"Final feature dimension: {self.final_feat_dim}")
        
        # Initialize feature encoding
        if encoding == 'learnable':
            self.init_learnable_encoding()
        elif encoding == 'bovw':
            self.init_bovw_encoding(vocab_size)
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")
        
        # Classification layer
        self.classifier = nn.Linear(self.final_feat_dim, num_classes)
        
    def init_learnable_encoding(self):
        """Initialize learnable feature encoding"""
        # Simple projection layer before pooling
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU()
        )
    
    def init_bovw_encoding(self, vocab_size: int):
        """Initialize Bag of Visual Words encoding"""
        self.vocab_size = vocab_size
        self.kmeans = None  # Will be fitted during training
        self.register_buffer('vocabulary', torch.zeros(vocab_size, self.feature_dim))
        
    def extract_features(self, img: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Extract local features from a single image
        Args:
            img: Input image in uint8 format
        Returns:
            Tuple of (descriptors, keypoints)
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
            
        # Extract keypoints and descriptors
        keypoints, descriptors = self.feature_extractor.detectAndCompute(img_gray, None)
        
        # Handle case when no features are detected
        if descriptors is None:
            descriptors = np.zeros((1, self.feature_dim), dtype=np.float32)
            keypoints = []
        else:
            # Ensure float32 type for descriptors
            descriptors = descriptors.astype(np.float32)
        
        return descriptors, keypoints

    def extract_features_batch(self, x: torch.Tensor) -> List[Tuple[np.ndarray, List]]:
        """
        Extract features from a batch of images
        Args:
            x: Batch of images (B, C, H, W)
        Returns:
            List of (descriptors, keypoints) tuples
        """
        features_list = []
        
        # Process each image in batch
        for img in x.cpu().numpy().transpose(0, 2, 3, 1):
            # Convert to uint8
            img = (img * 255).astype(np.uint8)
            
            # Extract features
            descriptors, keypoints = self.extract_features(img)
            features_list.append((descriptors, keypoints))
        
        return features_list
    
    def encode_features_learnable(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode features using projection and max pooling
        Args:
            features: Tensor of shape (B, N, D)
            mask: Optional mask for valid features (B, N)
        Returns:
            Encoded features of shape (B, 512)
        """
        batch_size = features.size(0)
        
        # Project features
        features = self.projector(features.view(-1, self.feature_dim))  # (B*N, 512)
        features = features.view(batch_size, -1, 512)  # (B, N, 512)
        
        # Apply mask if provided
        if mask is not None:
            features = features.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        # Max pooling over features
        pooled_features = torch.max(features, dim=1)[0]  # (B, 512)
        
        return pooled_features
    
    def encode_features_bovw(self, descriptors: np.ndarray, keypoints: List, image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Encode features using Bag of Visual Words with spatial pyramid
        Args:
            descriptors: Feature descriptors (N, D)
            keypoints: List of keypoints
            image_shape: Image dimensions (H, W)
        Returns:
            Encoded histogram
        """
        if len(descriptors) == 0:
            # Handle case with no features
            total_bins = sum(level * level for level in self.pyramid_levels)
            return torch.zeros(self.vocab_size * total_bins, dtype=torch.float32)
        
        # Convert descriptors to float64 for kmeans
        descriptors = descriptors.astype(np.float64)
        
        # Get feature assignments
        if self.kmeans is None:
            warnings.warn("Vocabulary not initialized. Using random assignments.")
            assignments = np.random.randint(0, self.vocab_size, size=len(descriptors))
        else:
            assignments = self.kmeans.predict(descriptors)
        
        # Get keypoint locations (normalized)
        if len(keypoints) > 0:
            kp_locs = np.array([(kp.pt[0] / image_shape[1], kp.pt[1] / image_shape[0]) 
                               for kp in keypoints])
        else:
            kp_locs = np.zeros((0, 2))
        
        # Initialize histograms for all levels
        histograms = []
        
        # Compute histogram for each pyramid level
        for level in self.pyramid_levels:
            bins_per_level = level * level
            weight = 2.0 ** (level - 1) if level > 1 else 1.0
            
            # Compute histogram for each spatial bin
            for i in range(level):
                for j in range(level):
                    # Get features in this bin
                    if len(kp_locs) > 0:
                        bin_mask = ((kp_locs[:, 0] >= i/level) & 
                                  (kp_locs[:, 0] < (i+1)/level) &
                                  (kp_locs[:, 1] >= j/level) & 
                                  (kp_locs[:, 1] < (j+1)/level))
                        bin_assignments = assignments[bin_mask]
                    else:
                        bin_assignments = np.array([], dtype=np.int64)
                    
                    # Compute histogram for this bin
                    if len(bin_assignments) > 0:
                        hist = np.bincount(bin_assignments, minlength=self.vocab_size)
                    else:
                        hist = np.zeros(self.vocab_size)
                    
                    hist = hist.astype(np.float32)
                    
                    # Normalize histogram
                    hist_sum = hist.sum()
                    if hist_sum > 0:
                        hist /= hist_sum
                    
                    histograms.append(hist * weight)
        
        # Concatenate all histograms
        final_histogram = np.concatenate(histograms)
        return torch.from_numpy(final_histogram).float()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        batch_size = x.size(0)
        device = x.device
        
        # Extract features from batch
        features_list = self.extract_features_batch(x)
        
        if self.encoding == 'learnable':
            # Prepare features and masks for learnable encoding
            padded_features = []
            attention_masks = []
            
            for descriptors, _ in features_list:
                n_features = len(descriptors)
                
                # Pad or sample features
                if n_features > self.num_features:
                    indices = np.random.choice(n_features, self.num_features, replace=False)
                    descriptors = descriptors[indices]
                    mask = torch.ones(self.num_features, dtype=torch.bool)
                else:
                    padding = np.zeros((self.num_features - n_features, self.feature_dim))
                    descriptors = np.vstack([descriptors, padding])
                    mask = torch.zeros(self.num_features, dtype=torch.bool)
                    mask[:n_features] = True
                
                padded_features.append(descriptors)
                attention_masks.append(mask)
            
            # Convert to tensors
            features = torch.from_numpy(np.stack(padded_features)).float().to(device)
            masks = torch.stack(attention_masks).to(device)
            
            # Encode features
            encoded_features = self.encode_features_learnable(features, masks)
            
        else:  # bovw
            encoded_features = []
            for img, (descriptors, keypoints) in zip(x.cpu().numpy().transpose(0, 2, 3, 1), features_list):
                hist = self.encode_features_bovw(descriptors, keypoints, img.shape[:2])
                encoded_features.append(hist)
            
            encoded_features = torch.stack(encoded_features).to(device)
        
        # Classification
        logits = self.classifier(encoded_features)
        
        return logits, encoded_features
    
    def fit_vocabulary(self, features: np.ndarray):
        """Fit k-means vocabulary for BoVW encoding"""
        if self.encoding != 'bovw':
            return
            
        # Convert features to float64 for kmeans
        features = features.astype(np.float64)
        
        print("Fitting vocabulary...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            batch_size=1024,
            random_state=42
        ).fit(features)
        
        # Store vocabulary in buffer (convert back to float32 for PyTorch)
        self.vocabulary.copy_(torch.from_numpy(self.kmeans.cluster_centers_.astype(np.float32)))
        print("Vocabulary fitted.") 