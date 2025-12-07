"""
Advanced LSTM Model with Attention Mechanism for sign language recognition.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple

from config import MODEL_CONFIG, FEATURE_CONFIG


class AttentionLayer(layers.Layer):
    """
    Multi-head attention mechanism for sequence modeling.
    """
    
    def __init__(self, num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=input_shape[-1] // self.num_heads
        )
        super().build(input_shape)
        
    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        return attention_output
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config


class AdvancedSignLanguageModel:
    """
    Advanced model architecture with Bi-LSTM and Attention.
    
    Architecture:
    1. Conv1D for local feature extraction
    2. Bi-LSTM for temporal modeling
    3. Multi-head Attention
    4. Dense layers with dropout
    5. Softmax classification
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize model builder.
        
        Args:
            config: Model configuration (uses MODEL_CONFIG if None)
        """
        self.config = config if config is not None else MODEL_CONFIG
        
    def build_model(self, num_classes: int, sequence_length: int) -> Model:
        """
        Build the complete model.
        
        Args:
            num_classes: Number of sign classes
            sequence_length: Length of input sequences
            
        Returns:
            Compiled Keras model
        """
        # Input
        inputs = keras.Input(
            shape=(sequence_length, FEATURE_CONFIG['total_features']),
            name='sequence_input'
        )
        
        # 1. Convolutional layer for local feature extraction
        x = layers.Conv1D(
            filters=self.config['conv_filters'],
            kernel_size=self.config['conv_kernel'],
            activation='relu',
            padding='same',
            name='conv1d'
        )(inputs)
        x = layers.BatchNormalization(name='bn_conv')(x)
        x = layers.Dropout(self.config['dropout_rate'] * 0.5, name='dropout_conv')(x)
        
        # 2. Bi-directional LSTM
        if self.config['use_bidirectional']:
            x = layers.Bidirectional(
                layers.LSTM(
                    self.config['lstm_units'],
                    return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                    name='lstm'
                ),
                name='bidirectional_lstm'
            )(x)
        else:
            x = layers.LSTM(
                self.config['lstm_units'],
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                name='lstm'
            )(x)
        
        x = layers.BatchNormalization(name='bn_lstm')(x)
        x = layers.Dropout(self.config['dropout_rate'], name='dropout_lstm')(x)
        
        # 3. Attention mechanism
        if self.config['use_attention']:
            x = AttentionLayer(
                num_heads=self.config['num_attention_heads'],
                name='attention'
            )(x)
            x = layers.BatchNormalization(name='bn_attention')(x)
        
        # 4. Global pooling
        x = layers.GlobalAveragePooling1D(name='global_pooling')(x)
        
        # 5. Dense layers
        x = layers.Dense(
            self.config['dense_units'],
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
            name='dense1'
        )(x)
        x = layers.BatchNormalization(name='bn_dense1')(x)
        x = layers.Dropout(self.config['dropout_rate'], name='dropout_dense1')(x)
        
        x = layers.Dense(
            self.config['dense_units'] // 2,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
            name='dense2'
        )(x)
        x = layers.BatchNormalization(name='bn_dense2')(x)
        x = layers.Dropout(self.config['dropout_rate'] * 0.5, name='dropout_dense2')(x)
        
        # 6. Output layer
        outputs = layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='SignLanguageModel')
        
        # Compile
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=self.config['gradient_clip_norm']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        return model
    
    def get_model_summary(self, model: Model) -> str:
        """Get model architecture summary."""
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)


# Crear instancia por defecto
def create_model(num_classes: int, sequence_length: int = 30) -> Model:
    """
    Convenience function to create model.
    
    Args:
        num_classes: Number of sign classes
        sequence_length: Sequence length (default: 30)
        
    Returns:
        Compiled model
    """
    builder = AdvancedSignLanguageModel()
    return builder.build_model(num_classes, sequence_length)