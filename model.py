"""
LungConVT: A Hybrid CNN-Transformer Architecture for Lung Disease Classification

This module implements the LungConVT architecture, which combines convolutional
neural networks with transformer blocks for medical image classification.

Architecture Components:
- Depthwise Separable Convolutions (DC layers)
- Dual-Head Convolutional Multi-Head Attention (DHC-MHA)
- Adaptive Multi-Grained Multi-Head Attention (AMG-MHA)

Reference:
[Add your paper reference here when published]

Author: [Your Name]
Date: [Current Date]
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, 
    GlobalAveragePooling2D, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Add, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LungConVTModel:
    """
    LungConVT: Hybrid CNN-Transformer for lung disease classification.
    
    This architecture combines the local feature extraction capabilities
    of CNNs with the global context modeling of transformers.
    """
    
    @staticmethod
    def mlp(x, hidden_units, dropout_rate):
        """
        Multi-Layer Perceptron block.
        
        Args:
            x: Input tensor
            hidden_units: List of hidden layer dimensions
            dropout_rate: Dropout rate
            
        Returns:
            Output tensor
        """
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.swish)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    @staticmethod
    def dhc_mha(x, num_heads, ff_dim, dropout_rate):
        """
        Dual-Head Convolutional Multi-Head Attention block.
        
        Combines multi-head attention with convolutional feed-forward layers
        for better local-global feature integration.
        
        Args:
            x: Input tensor
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout_rate: Dropout rate
            
        Returns:
            Output tensor
        """
        # Layer normalization
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Multi-head attention
        key_dim = x.shape[-1] // num_heads
        attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim
        )(x1, x1)
        
        # Residual connection
        x2 = layers.Add()([x, attention])
        x3 = layers.LayerNormalization()(x2)
        
        # Convolutional feed-forward network
        x4 = layers.Conv2D(
            filters=ff_dim, 
            kernel_size=(3, 3), 
            activation=tf.nn.swish, 
            padding='same'
        )(x3)
        x4 = layers.Dropout(dropout_rate)(x4)
        
        x4 = layers.Conv2D(
            filters=x.shape[-1], 
            kernel_size=(3, 3), 
            activation=tf.nn.swish, 
            padding='same'
        )(x4)
        x4 = layers.Dropout(dropout_rate)(x4)
        
        # Final residual connection
        x = layers.Add()([x3, x4])
        x = layers.LayerNormalization()(x)
        
        return x
    
    @staticmethod
    def amg_mha(x, transformer_layers, projection_dim, num_heads=2):
        """
        Adaptive Multi-Grained Multi-Head Attention block.
        
        Processes features at multiple granularities using transformer layers.
        
        Args:
            x: Input tensor
            transformer_layers: Number of transformer layers
            projection_dim: Projection dimension
            num_heads: Number of attention heads
            
        Returns:
            Output tensor
        """
        for _ in range(transformer_layers):
            # Layer normalization
            x1 = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=projection_dim, 
                dropout=0.1
            )(x1, x1)
            
            # Residual connection
            x2 = layers.Add()([attention_output, x])
            
            # MLP block
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = LungConVTModel.mlp(
                x3, 
                hidden_units=[x.shape[-1] * 2, x.shape[-1]], 
                dropout_rate=0.1
            )
            
            # Final residual connection
            x = layers.Add()([x3, x2])
        
        return x
    
    @staticmethod
    def depth_block(x, strides=(1, 1)):
        """
        Depthwise separable convolution block.
        
        Args:
            x: Input tensor
            strides: Stride for convolution
            
        Returns:
            Output tensor
        """
        x = DepthwiseConv2D(
            3, 
            strides=strides, 
            padding='same', 
            use_bias=False
        )(x)
        x = BatchNormalization()(x)
        x = tf.nn.swish(x)
        return x
    
    @staticmethod
    def single_conv_block(x, filters):
        """
        Single convolution block with batch normalization and activation.
        
        Args:
            x: Input tensor
            filters: Number of filters
            
        Returns:
            Output tensor
        """
        x = Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = tf.nn.swish(x)
        return x
    
    @staticmethod
    def dc_layer(x, filters, strides):
        """
        Depthwise Convolution layer with residual connection.
        
        Combines depthwise separable convolution with residual connections
        for efficient feature extraction.
        
        Args:
            x: Input tensor
            filters: Number of output filters
            strides: Stride for convolution
            
        Returns:
            Output tensor
        """
        shortcut = x
        
        # Depthwise separable convolution
        x = LungConVTModel.depth_block(x, strides)
        x = LungConVTModel.single_conv_block(x, filters)
        
        # Adjust shortcut dimensions if needed
        shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        
        # Residual connection
        x = layers.Add()([x, shortcut])
        
        return x
    
    @staticmethod
    def build_model(input_shape=(256, 256, 3), n_classes=4):
        """
        Build the complete LungConVT model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            n_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building LungConVT model with input shape {input_shape}")
        
        # Input layer
        input_layer = Input(input_shape)
        
        # Initial convolution blocks
        x = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False)(input_layer)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(32, 3, strides=(1, 1), padding='same', use_bias=False)(x)
        
        # First stage: Low-level features
        x = LungConVTModel.dc_layer(x, 32, strides=(1, 1))
        x = LungConVTModel.dc_layer(x, 64, strides=(2, 2))
        
        # Second stage: Mid-level features
        x = LungConVTModel.dc_layer(x, 64, strides=(1, 1))
        x = LungConVTModel.dc_layer(x, 128, strides=(2, 2))
        
        # Third stage: High-level features with attention
        x = LungConVTModel.dhc_mha(x, num_heads=4, ff_dim=128, dropout_rate=0.2)
        x = LungConVTModel.dc_layer(x, 128, strides=(2, 2))
        x = LungConVTModel.dhc_mha(x, num_heads=8, ff_dim=128, dropout_rate=0.2)
        
        logger.info(f"Feature map shape before reshape: {x.shape}")
        
        # Reshape for multi-grained attention
        num_patches = int((x.shape[1] * x.shape[2]) / 4)
        logger.info(f"Number of patches: {num_patches}")
        
        x = layers.Reshape((4, num_patches, 128))(x)
        logger.info(f"Reshaped tensor shape: {x.shape}")
        
        # Fourth stage: Multi-grained attention
        x = LungConVTModel.amg_mha(x, 2, 64, 4)
        x = LungConVTModel.dc_layer(x, 128, strides=(1, 1))
        x = LungConVTModel.amg_mha(x, 2, 128, 8)
        
        # Final stage
        x = LungConVTModel.dc_layer(x, 256, strides=(1, 1))
        
        # Global pooling and classification
        x = GlobalAveragePooling2D()(x)
        output = Dense(n_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output, name='LungConVT')
        
        logger.info(f"Model created with {model.count_params():,} parameters")
        
        return model


def create_lungconvt_model(input_shape=(256, 256, 3), n_classes=4):
    """
    Factory function to create a LungConVT model.
    
    Args:
        input_shape: Input image shape
        n_classes: Number of output classes
        
    Returns:
        Uncompiled Keras model
    """
    return LungConVTModel.build_model(input_shape, n_classes)


if __name__ == "__main__":
    # Example usage
    model = create_lungconvt_model(input_shape=(256, 256, 3), n_classes=4)
    model.summary()