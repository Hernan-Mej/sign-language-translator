"""
Data Augmentation for sign language sequences.
Includes temporal, spatial, and noise-based augmentation techniques.
"""

import numpy as np
from typing import List, Tuple
from scipy import interpolate

from config import AUGMENTATION_CONFIG


class SignLanguageAugmenter:
    """
    Comprehensive data augmentation for sign language sequences.
    
    Techniques:
    - Temporal: speed variation, random cropping, time warping
    - Spatial: scale, translation, rotation
    - Noise: gaussian noise, feature dropout
    - Mirror: horizontal flip
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize augmenter with configuration.
        
        Args:
            config: Augmentation configuration (uses AUGMENTATION_CONFIG if None)
        """
        self.config = config if config is not None else AUGMENTATION_CONFIG
        
    def augment(self, sequence: np.ndarray, label: int) -> List[Tuple[np.ndarray, int]]:
        """
        Apply augmentation to a sequence.
        
        Args:
            sequence: Input sequence (frames, features)
            label: Class label
            
        Returns:
            List of (augmented_sequence, label) tuples
        """
        if not self.config['enabled']:
            return [(sequence, label)]
        
        augmented = [(sequence.copy(), label)]  # Original
        
        factor = self.config['augmentation_factor']
        
        for _ in range(factor - 1):
            aug_seq = sequence.copy()
            
            # Apply augmentations probabilistically
            if np.random.rand() < self.config['augmentation_probability']:
                
                # Temporal augmentation
                if self.config['temporal_enabled'] and np.random.rand() < 0.5:
                    aug_method = np.random.choice(['speed', 'crop'])
                    if aug_method == 'speed':
                        aug_seq = self.time_warp(aug_seq)
                    else:
                        aug_seq = self.random_crop(aug_seq)
                
                # Spatial augmentation
                if self.config['spatial_enabled'] and np.random.rand() < 0.5:
                    aug_seq = self.spatial_transform(aug_seq)
                
                # Noise augmentation
                if self.config['noise_enabled'] and np.random.rand() < 0.3:
                    aug_method = np.random.choice(['gaussian', 'dropout'])
                    if aug_method == 'gaussian':
                        aug_seq = self.add_gaussian_noise(aug_seq)
                    else:
                        aug_seq = self.feature_dropout(aug_seq)
                
                # Mirror augmentation
                if self.config['mirror_enabled'] and np.random.rand() < self.config['mirror_probability']:
                    aug_seq = self.mirror_augment(aug_seq)
            
            augmented.append((aug_seq, label))
        
        return augmented
    
    def time_warp(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply time warping (speed variation).
        
        Args:
            sequence: Input sequence (frames, features)
            
        Returns:
            Time-warped sequence
        """
        speed_min, speed_max = self.config['speed_range']
        speed_factor = np.random.uniform(speed_min, speed_max)
        
        original_length = len(sequence)
        new_length = int(original_length / speed_factor)
        
        if new_length < 5:  # Minimum sequence length
            return sequence
        
        # Interpolate
        x_old = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, new_length)
        
        warped = np.zeros((new_length, sequence.shape[1]))
        
        for i in range(sequence.shape[1]):
            f = interpolate.interp1d(x_old, sequence[:, i], kind='linear', fill_value='extrapolate')
            warped[:, i] = f(x_new)
        
        return warped
    
    def random_crop(self, sequence: np.ndarray) -> np.ndarray:
        """
        Randomly crop the sequence temporally.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Cropped sequence
        """
        crop_min, crop_max = self.config['crop_ratio_range']
        crop_ratio = np.random.uniform(crop_min, crop_max)
        
        original_length = len(sequence)
        new_length = max(5, int(original_length * crop_ratio))  # Minimum 5 frames
        
        if new_length >= original_length:
            return sequence
        
        start_idx = np.random.randint(0, original_length - new_length + 1)
        end_idx = start_idx + new_length
        
        return sequence[start_idx:end_idx].copy()
    
    def spatial_transform(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply spatial transformations (scale, translation, rotation).
        
        Args:
            sequence: Input sequence
            
        Returns:
            Transformed sequence
        """
        transformed = sequence.copy()
        
        # Extract spatial features (first 63: x, y, z coordinates)
        spatial_features = transformed[:, :63].reshape(-1, 21, 3)
        
        # Scale
        scale_min, scale_max = self.config['scale_range']
        scale = np.random.uniform(scale_min, scale_max)
        spatial_features *= scale
        
        # Translation
        trans_range = self.config['translation_range']
        translation = np.random.uniform(-trans_range, trans_range, size=3)
        spatial_features += translation
        
        # Rotation (around z-axis)
        rot_range = self.config['rotation_range']
        angle = np.radians(np.random.uniform(-rot_range, rot_range))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        for frame_idx in range(spatial_features.shape[0]):
            spatial_features[frame_idx] = spatial_features[frame_idx] @ rotation_matrix.T
        
        # Update sequence
        transformed[:, :63] = spatial_features.reshape(-1, 63)
        
        return transformed
    
    def add_gaussian_noise(self, sequence: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to features.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Noisy sequence
        """
        noise_std = self.config['gaussian_noise_std']
        noise = np.random.normal(0, noise_std, sequence.shape)
        return sequence + noise
    
    def feature_dropout(self, sequence: np.ndarray) -> np.ndarray:
        """
        Randomly drop out features.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Sequence with dropped features
        """
        dropout_prob = self.config['feature_dropout_prob']
        mask = np.random.binomial(1, 1 - dropout_prob, sequence.shape)
        return sequence * mask
    
    def mirror_augment(self, sequence: np.ndarray) -> np.ndarray:
        """
        Mirror the hand horizontally.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Mirrored sequence
        """
        mirrored = sequence.copy()
        
        # Mirror x-coordinates (first 63 features, every 3rd starting from 0)
        spatial_features = mirrored[:, :63].reshape(-1, 21, 3)
        spatial_features[:, :, 0] *= -1  # Flip x-axis
        mirrored[:, :63] = spatial_features.reshape(-1, 63)
        
        # Mirror handedness (last feature: 0=left, 1=right)
        if mirrored.shape[1] >= 240:  # Ensure we have the handedness feature
            mirrored[:, -1] = 1.0 - mirrored[:, -1]
        
        return mirrored
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment entire dataset.
        
        Args:
            X: Input sequences (n_samples, variable_length, n_features)
            y: Labels (n_samples,)
            
        Returns:
            Augmented (X, y)
        """
        if not self.config['enabled']:
            return X, y
        
        X_aug, y_aug = [], []
        
        for i in range(len(X)):
            augmented_samples = self.augment(X[i], y[i])
            for aug_seq, aug_label in augmented_samples:
                X_aug.append(aug_seq)
                y_aug.append(aug_label)
        
        return np.array(X_aug, dtype=object), np.array(y_aug)


# Example usage
if __name__ == "__main__":
    # Create augmenter
    augmenter = SignLanguageAugmenter()
    
    # Example sequence (30 frames, 240 features)
    sequence = np.random.randn(30, 240)
    label = 0
    
    # Augment
    augmented_samples = augmenter.augment(sequence, label)
    
    print(f"Original: {sequence.shape}")
    print(f"Augmented: {len(augmented_samples)} samples")
    for i, (aug_seq, aug_label) in enumerate(augmented_samples):
        print(f"  Sample {i}: {aug_seq.shape}, label={aug_label}")