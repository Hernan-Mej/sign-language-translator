"""
Training pipeline for sign language recognition model.
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

from config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR,
    TRAINING_CONFIG, SEQUENCE_CONFIG,
    MODEL_PATHS
)
from key_points_extractor import KeyPointsExtractor
from data_augmentation import SignLanguageAugmenter
from advanced_lstm_model import AdvancedSignLanguageModel


class Trainer:
    """
    Complete training pipeline for sign language model.
    """
    
    def __init__(self, 
                 data_dir: Path = None,
                 models_dir: Path = None,
                 logs_dir: Path = None,
                 use_augmentation: bool = True):
        """
        Initialize trainer.
        
        Args:
            data_dir: Data directory path
            models_dir: Models directory path
            logs_dir: Logs directory path
            use_augmentation: Enable data augmentation
        """
        self.data_dir = data_dir or DATA_DIR
        self.models_dir = models_dir or MODELS_DIR
        self.logs_dir = logs_dir or LOGS_DIR
        self.use_augmentation = use_augmentation
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize augmenter
        self.augmenter = SignLanguageAugmenter() if use_augmentation else None
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load and prepare data from raw directory.
        
        Returns:
            (X, y, sign_map) where:
            - X: sequences (n_samples, variable_length, 240)
            - y: labels (n_samples,)
            - sign_map: {class_id: sign_name}
        """
        print("📂 Loading data from:", self.data_dir / "raw")
        
        X, y = [], []
        sign_map = {}
        class_id = 0
        
        raw_dir = self.data_dir / "raw"
        if not raw_dir.exists():
            raise ValueError(f"Data directory not found: {raw_dir}")
        
        sign_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
        
        for sign_dir in sign_dirs:
            sign_name = sign_dir.name
            sign_map[class_id] = sign_name
            
            sample_files = list(sign_dir.glob("*.npy"))
            print(f"  {sign_name}: {len(sample_files)} samples")
            
            for sample_file in sample_files:
                try:
                    sequence = np.load(sample_file)
                    
                    # Validate
                    if sequence.shape[1] != 240:
                        print(f"    Warning: Invalid shape {sequence.shape} in {sample_file}")
                        continue
                    
                    X.append(sequence)
                    y.append(class_id)
                    
                except Exception as e:
                    print(f"    Error loading {sample_file}: {e}")
            
            class_id += 1
        
        print(f"\n✓ Loaded {len(X)} samples from {len(sign_map)} classes")
        
        # Save sign map
        sign_map_path = self.data_dir / "sign_map.json"
        with open(sign_map_path, 'w') as f:
            json.dump(sign_map, f, indent=2)
        
        return np.array(X, dtype=object), np.array(y), sign_map
    
    def normalize_sequence_lengths(self, 
                                   X: np.ndarray, 
                                   target_length: int = None) -> np.ndarray:
        """
        Normalize all sequences to same length.
        
        Args:
            X: Variable-length sequences
            target_length: Target length (None = median)
            
        Returns:
            Fixed-length sequences (n_samples, target_length, 240)
        """
        if target_length is None:
            lengths = [len(seq) for seq in X]
            target_length = int(np.median(lengths))
        
        print(f"🔧 Normalizing sequences to length: {target_length}")
        
        X_normalized = []
        
        for sequence in X:
            current_length = len(sequence)
            
            if current_length < target_length:
                # Pad
                padding = np.zeros((target_length - current_length, 240))
                normalized = np.vstack([sequence, padding])
            elif current_length > target_length:
                # Truncate or interpolate
                indices = np.linspace(0, current_length - 1, target_length).astype(int)
                normalized = sequence[indices]
            else:
                normalized = sequence
            
            X_normalized.append(normalized)
        
        return np.array(X_normalized)
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   stratify: bool = True) -> Tuple:
        """
        Split data into train/val/test sets.
        
        Args:
            X: Input sequences
            y: Labels
            stratify: Use stratified split
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_split = TRAINING_CONFIG['train_split']
        val_split = TRAINING_CONFIG['val_split']
        test_split = TRAINING_CONFIG['test_split']
        
        # First split: train + (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=train_split,
            random_state=TRAINING_CONFIG['random_seed'],
            stratify=y if stratify else None,
            shuffle=TRAINING_CONFIG['shuffle']
        )
        
        # Second split: val + test
        val_ratio = val_split / (val_split + test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_ratio,
            random_state=TRAINING_CONFIG['random_seed'],
            stratify=y_temp if stratify else None,
            shuffle=TRAINING_CONFIG['shuffle']
        )
        
        print(f"\n📊 Data split:")
        print(f"  Train: {len(X_train)} samples ({train_split:.0%})")
        print(f"  Val:   {len(X_val)} samples ({val_split:.0%})")
        print(f"  Test:  {len(X_test)} samples ({test_split:.0%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to training set.
        
        Args:
            X: Input sequences
            y: Labels
            
        Returns:
            Augmented (X, y)
        """
        if not self.use_augmentation or self.augmenter is None:
            return X, y
        
        print("\n🔄 Applying data augmentation...")
        X_aug, y_aug = self.augmenter.augment_dataset(X, y)
        
        print(f"  Original: {len(X)} samples")
        print(f"  Augmented: {len(X_aug)} samples")
        print(f"  Increase: {len(X_aug) / len(X):.1f}x")
        
        return X_aug, y_aug
    
    def compute_class_weights(self, y: np.ndarray) -> Dict:
        """Compute class weights for imbalanced datasets."""
        if not TRAINING_CONFIG['use_class_weights']:
            return None
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def train_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   num_classes: int,
                   model_name: str = "sign_language_model") -> Tuple[keras.Model, Dict]:
        """
        Train the model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            num_classes: Number of classes
            model_name: Model name for saving
            
        Returns:
            (trained_model, history_dict)
        """
        print("\n🏗️ Building model...")
        
        # Build model
        model_builder = AdvancedSignLanguageModel()
        model = model_builder.build_model(
            num_classes=num_classes,
            sequence_length=X_train.shape[1]
        )
        
        print(model_builder.get_model_summary(model))
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(self.models_dir / f"{model_name}_best.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(self.logs_dir / "tensorboard"),
                histogram_freq=1
            )
        ]
        
        # Class weights
        class_weights = self.compute_class_weights(y_train)
        
        # Train
        print("\n🚀 Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=16,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model_path = self.models_dir / f"{model_name}.h5"
        model.save(str(model_path))
        print(f"\n💾 Model saved: {model_path}")
        
        # Save history
        history_path = self.models_dir / f"{model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2)
        
        return model, history.history
    
    def evaluate_model(self,
                      model: keras.Model,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      sign_map: Dict) -> Dict:
        """
        Evaluate trained model.
        
        Args:
            model: Trained model
            X_test, y_test: Test data
            sign_map: Class mapping
            
        Returns:
            Evaluation metrics
        """
        print("\n📈 Evaluating model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        accuracy = np.mean(y_pred_classes == y_test)
        
        print(f"\n✅ Test Accuracy: {accuracy:.2%}")
        
        print("\n📊 Classification Report:")
        print(classification_report(
            y_test, y_pred_classes,
            target_names=[sign_map[i] for i in range(len(sign_map))]
        ))
        
        return {
            'metrics': {
                'accuracy': accuracy,
                'top_3_accuracy': self._top_k_accuracy(y_test, y_pred, k=3)
            },
            'predictions': y_pred_classes,
            'confusion_matrix': confusion_matrix(y_test, y_pred_classes).tolist()
        }
    
    @staticmethod
    def _top_k_accuracy(y_true, y_pred, k=3):
        """Calculate top-k accuracy."""
        top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
        return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])