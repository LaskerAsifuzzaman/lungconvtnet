"""
Training module for LungConVT model.

This module handles model training, including optimizer configuration,
callbacks setup, and training loop execution.

Author: [Your Name]
Date: [Current Date]
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
)
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class ModelTrainer:
    """
    Trainer class for LungConVT model.
    
    Handles training configuration, execution, and monitoring.
    """
    
    def __init__(self, model, model_name=None, output_dir='outputs'):
        """
        Initialize the trainer.
        
        Args:
            model: Keras model to train
            model_name: Name for saving model and logs
            output_dir: Directory for saving outputs
        """
        self.model = model
        self.model_name = model_name or f"lungconvt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = output_dir
        
        # Create output directories
        self._create_output_dirs()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
    def _create_output_dirs(self):
        """Create necessary output directories."""
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        
        for dir_path in [self.model_dir, self.log_dir, self.checkpoint_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
    def _set_seeds(self, seed=42):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seeds set to {seed}")
        
    def compile_model(self, learning_rate=0.001, optimizer_params=None):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Initial learning rate
            optimizer_params: Additional optimizer parameters
        """
        if optimizer_params is None:
            optimizer_params = {
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-08
            }
            
        optimizer = Adam(learning_rate=learning_rate, **optimizer_params)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model compiled with learning rate: {learning_rate}")
        
    def get_callbacks(self):
        """
        Create training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=7,
            cooldown=5,
            min_lr=1e-10,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint - save best model
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.model_name}_best.h5"
        )
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        )
        callbacks.append(model_checkpoint)
        
        # CSV logger for training history
        csv_path = os.path.join(self.log_dir, f"{self.model_name}_history.csv")
        csv_logger = CSVLogger(filename=csv_path, separator=',', append=False)
        callbacks.append(csv_logger)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        logger.info(f"Created {len(callbacks)} callbacks")
        return callbacks
        
    def train(self, train_generator, val_generator, epochs=200, 
              class_weights=None, verbose=1):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of training epochs
            class_weights: Optional class weights for imbalanced data
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Save training configuration
        self._save_training_config(epochs, train_generator.batch_size)
        
        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=verbose,
            workers=4,
            use_multiprocessing=False
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(training_time))}")
        
        # Save final model
        self._save_final_model()
        
        # Save training summary
        self._save_training_summary(history, training_time)
        
        return history
        
    def _save_training_config(self, epochs, batch_size):
        """Save training configuration."""
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        config = {
            'model_name': self.model_name,
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'optimizer': self.model.optimizer.get_config(),
            'loss': self.model.loss,
            'metrics': self.model.metrics_names,
            'model_params': int(self.model.count_params()),
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.log_dir, f"{self.model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4, cls=NumpyEncoder)
            
    def _save_final_model(self):
        """Save the final trained model."""
        model_path = os.path.join(self.model_dir, f"{self.model_name}_final.h5")
        self.model.save(model_path)
        logger.info(f"Final model saved to {model_path}")
        
    def _save_training_summary(self, history, training_time):
        """Save training summary."""
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        summary = {
            'model_name': self.model_name,
            'training_time_seconds': float(training_time),
            'training_time_formatted': time.strftime('%H:%M:%S', time.gmtime(training_time)),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_val_accuracy_epoch': int(np.argmax(history.history['val_accuracy']) + 1),
            'total_epochs': len(history.history['loss'])
        }
        
        summary_path = os.path.join(self.log_dir, f"{self.model_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4, cls=NumpyEncoder)
            
        logger.info(f"Training summary saved to {summary_path}")
        
    def load_best_model(self):
        """Load the best model from checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.model_name}_best.h5"
        )
        
        if os.path.exists(checkpoint_path):
            self.model = tf.keras.models.load_model(checkpoint_path)
            logger.info(f"Loaded best model from {checkpoint_path}")
            return self.model
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return None


def calculate_class_weights(train_generator):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        train_generator: Training data generator
        
    Returns:
        dict: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get all labels
    labels = []
    for i in range(len(train_generator)):
        _, batch_labels = train_generator[i]
        labels.extend(np.argmax(batch_labels, axis=1))
    
    # Compute class weights
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=labels
    )
    
    # Convert to regular Python types to avoid JSON serialization issues
    class_weight_dict = {}
    for i, weight in enumerate(class_weights):
        class_weight_dict[int(unique_classes[i])] = float(weight)
    
    logger.info(f"Calculated class weights: {class_weight_dict}")
    
    return class_weight_dict


if __name__ == "__main__":
    # Example usage
    from model import create_lungconvt_model
    from data_loader import create_data_generators
    
    # Create data generators
    train_gen, val_gen, labels = create_data_generators(
        csv_path='data_mapping.csv',
        image_size=(256, 256),
        batch_size=16
    )
    
    # Create model
    model = create_lungconvt_model(input_shape=(256, 256, 3), n_classes=len(labels))
    
    # Create trainer
    trainer = ModelTrainer(model, model_name='lungconvt_experiment')
    
    # Compile model
    trainer.compile_model(learning_rate=0.001)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_gen)
    
    # Train model
    history = trainer.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=200,
        class_weights=class_weights
    )