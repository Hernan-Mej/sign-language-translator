"""
Configuration for Sign Language Translator.
Automatically detects Colab vs Local environment.
"""

import os
from pathlib import Path

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

def detect_environment():
    """Detect if running in Colab or local environment."""
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "local"

ENVIRONMENT = detect_environment()
IN_COLAB = (ENVIRONMENT == "colab")

# ============================================================================
# PROJECT PATHS
# ============================================================================

if IN_COLAB:
    # Colab: Use Google Drive for persistence
    DRIVE_ROOT = Path('/content/drive/MyDrive')
    PROJECT_NAME = 'SignLanguageTranslator'
    ROOT_DIR = DRIVE_ROOT / PROJECT_NAME
    
    # Code is cloned to /content/
    CODE_DIR = Path('/content/sign-language-translator')
else:
    # Local: Use project directory
    CODE_DIR = Path(__file__).parent.parent  # Go up from src/ to root
    ROOT_DIR = CODE_DIR

# Data directories (in Drive for Colab, local for local)
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR / "raw", exist_ok=True)
os.makedirs(DATA_DIR / "processed", exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Architecture
    "lstm_units": 256,
    "dense_units": 128,
    "dropout_rate": 0.4,
    "use_attention": True,
    "num_attention_heads": 8,
    "use_bidirectional": True,
    "conv_filters": 128,
    "conv_kernel": 3,
    
    # Training
    "learning_rate": 0.0005,
    "batch_size": 16,
    "epochs": 150,
    "early_stopping_patience": 20,
    
    # Regularization
    "l2_regularization": 0.0001,
    "gradient_clip_norm": 1.0,
    
    # Optimization
    "use_mixed_precision": False,
    "use_xla": False,
}

# ============================================================================
# FEATURE EXTRACTION CONFIGURATION
# ============================================================================

FEATURE_CONFIG = {
    # Keypoint extraction
    "use_3d_keypoints": True,
    "history_size": 5,
    
    # Feature dimensions
    "spatial_features": 63,      # 21 landmarks × 3 coords
    "geometric_features": 35,    # Angles, distances, lengths
    "shape_features": 15,        # Hand shape descriptors
    "temporal_features": 126,    # Velocity and acceleration
    "handedness_feature": 1,     # Left/right hand
    "total_features": 240,       # Total feature dimension
    
    # Normalization
    "normalize_translation": True,
    "normalize_rotation": True,
    "normalize_scale": True,
}

# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================

AUGMENTATION_CONFIG = {
    "enabled": True,
    "augmentation_probability": 0.5,
    "augmentation_factor": 2,
    
    # Temporal augmentation
    "temporal_enabled": True,
    "speed_range": (0.8, 1.2),
    "crop_ratio_range": (0.85, 1.0),
    
    # Spatial augmentation
    "spatial_enabled": True,
    "scale_range": (0.9, 1.1),
    "translation_range": 0.05,
    "rotation_range": 15.0,
    
    # Noise augmentation
    "noise_enabled": True,
    "gaussian_noise_std": 0.01,
    "feature_dropout_prob": 0.1,
    
    # Mirror augmentation
    "mirror_enabled": True,
    "mirror_probability": 0.3,
}

# ============================================================================
# VIDEO PROCESSING CONFIGURATION
# ============================================================================

VIDEO_CONFIG = {
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "keypoint_confidence_threshold": 0.5,
    "detection_confidence": 0.5,
}

# ============================================================================
# MEDIAPIPE CONFIGURATION
# ============================================================================

MEDIAPIPE_CONFIG = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "model_complexity": 1,
    "max_num_hands": 2,
}

# ============================================================================
# SEQUENCE PROCESSING CONFIGURATION
# ============================================================================

SEQUENCE_CONFIG = {
    "min_sequence_length": 10,
    "max_sequence_length": 60,
    "sequence_stride": 5,
    "padding_value": 0.0,
}

# ============================================================================
# TRAINING DATA CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
    "shuffle": True,
    
    # Class balancing
    "use_class_weights": True,
    "oversample_minority": False,
    "undersample_majority": False,
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

UI_CONFIG = {
    "theme": "soft",
    "show_keypoints": True,
    "show_confidence": True,
    "show_fps": True,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_to_file": True,
    "log_file": LOGS_DIR / "training.log",
    "tensorboard_enabled": True,
    "tensorboard_dir": LOGS_DIR / "tensorboard",
}

# ============================================================================
# DATASET PATHS
# ============================================================================

DATASET_PATHS = {
    "raw_data": DATA_DIR / "raw",
    "processed_data": DATA_DIR / "processed",
    "sign_map": DATA_DIR / "sign_map.json",
    "dataset_info": DATA_DIR / "dataset_info.json",
}

# ============================================================================
# MODEL PATHS
# ============================================================================

MODEL_PATHS = {
    "model": MODELS_DIR / "sign_language_model.h5",
    "best_model": MODELS_DIR / "sign_language_model_best.h5",
    "training_history": MODELS_DIR / "training_history.json",
    "model_config": MODELS_DIR / "model_config.json",
    "architecture_plot": MODELS_DIR / "model_architecture.png",
}

# ============================================================================
# PRINT CONFIGURATION INFO
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Sign Language Translator - Configuration")
    print("="*60)
    print(f"\nEnvironment: {ENVIRONMENT.upper()}")
    print(f"In Colab: {IN_COLAB}")
    print(f"\nPaths:")
    print(f"  ROOT_DIR:   {ROOT_DIR}")
    print(f"  DATA_DIR:   {DATA_DIR}")
    print(f"  MODELS_DIR: {MODELS_DIR}")
    print(f"  LOGS_DIR:   {LOGS_DIR}")
    print("\nModel Config:")
    print(f"  LSTM Units: {MODEL_CONFIG['lstm_units']}")
    print(f"  Features:   {FEATURE_CONFIG['total_features']}")
    print(f"  Attention:  {MODEL_CONFIG['use_attention']}")
    print(f"  Bi-LSTM:    {MODEL_CONFIG['use_bidirectional']}")
    print("\nAugmentation:")
    print(f"  Enabled:    {AUGMENTATION_CONFIG['enabled']}")
    print(f"  Factor:     {AUGMENTATION_CONFIG['augmentation_factor']}x")
    print("="*60)