"""
Utility functions for lung disease classification project.

This module contains helper functions for visualization, logging,
and other common operations.

Author: [Your Name]
Date: [Current Date]
"""

import os
import json
import yaml
import numpy as np

# Fix for matplotlib backend in headless environments
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import random
import tensorflow as tf
from PIL import Image


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


def set_global_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For CUDA determinism (may impact performance)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    logging.info(f"Global seeds set to {seed}")


def setup_logging(log_dir='outputs/logs', log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def plot_training_history(history, save_path=None, figsize=(12, 5)):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Keras training history object or dict
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert history object to dict if needed
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    epochs = range(1, len(history_dict['accuracy']) + 1)
    ax1.plot(epochs, history_dict['accuracy'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(epochs, history_dict['loss'], 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, history_dict['val_loss'], 'r-', label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training history plot saved to {save_path}")
    
    # Only show if not in headless mode
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    else:
        plt.close()


def create_data_distribution_plot(data_loader, save_path=None, figsize=(10, 6)):
    """
    Create a bar plot of class distribution.
    
    Args:
        data_loader: DataLoader instance with loaded data
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Get class distribution
    distribution = data_loader.get_class_distribution()
    
    # Create plot
    plt.figure(figsize=figsize)
    labels = list(distribution.keys())
    counts = list(distribution.values())
    
    bars = plt.bar(labels, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=12)
    
    plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Data distribution plot saved to {save_path}")
    
    # Only show if not in headless mode
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    else:
        plt.close()


def load_config(config_path):
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
    
    logging.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config, save_path):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    with open(save_path, 'w') as f:
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.endswith('.json'):
            json.dump(config, f, indent=4, cls=NumpyEncoder)
        else:
            raise ValueError(f"Unsupported config file format: {save_path}")
    
    logging.info(f"Configuration saved to {save_path}")


def create_sample_grid(data_generator, num_samples=16, save_path=None, figsize=(12, 12)):
    """
    Create a grid of sample images from the data generator.
    
    Args:
        data_generator: Data generator
        num_samples: Number of samples to display
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Get a batch of data
    images, labels = next(data_generator)
    
    # Limit to requested number of samples
    num_samples = min(num_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        img = images[i]
        
        # Get label (handle one-hot encoding)
        if len(labels.shape) > 1:
            label_idx = np.argmax(labels[i])
            label = data_generator.class_indices
            label = list(label.keys())[list(label.values()).index(label_idx)]
        else:
            label = str(labels[i])
        
        ax.imshow(img)
        ax.set_title(f'Class: {label}', fontsize=10)
        ax.axis('off')
    
    # Hide remaining axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Sample grid saved to {save_path}")
    
    # Only show if not in headless mode
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    else:
        plt.close()


def get_model_summary(model, save_path=None):
    """
    Get and save model summary.
    
    Args:
        model: Keras model
        save_path: Path to save summary
        
    Returns:
        str: Model summary
    """
    # Capture model summary
    from io import StringIO
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    model.summary()
    
    summary = buffer.getvalue()
    sys.stdout = old_stdout
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary)
        logging.info(f"Model summary saved to {save_path}")
    
    return summary


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: Array of labels
        
    Returns:
        dict: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Convert one-hot to labels if needed
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    
    # Compute weights
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=labels
    )
    
    class_weights = dict(zip(unique_classes, weights))
    
    logging.info(f"Computed class weights: {class_weights}")
    return class_weights


def get_system_info():
    """
    Get system and hardware information.
    
    Returns:
        dict: System information
    """
    import platform
    import psutil
    
    info = {
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'gpu_available': tf.config.list_physical_devices('GPU') != [],
        'gpu_devices': [gpu.name for gpu in tf.config.list_physical_devices('GPU')]
    }
    
    return info


def save_system_info(save_path):
    """
    Save system information to file.
    
    Args:
        save_path: Path to save system info
    """
    info = get_system_info()
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4, cls=NumpyEncoder)
    
    logging.info(f"System info saved to {save_path}")


def create_experiment_directory(base_dir='experiments', experiment_name=None):
    """
    Create a directory structure for an experiment.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        dict: Paths to created directories
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'experiment_{timestamp}'
    
    # Create directory structure
    experiment_dir = os.path.join(base_dir, experiment_name)
    subdirs = {
        'root': experiment_dir,
        'models': os.path.join(experiment_dir, 'models'),
        'logs': os.path.join(experiment_dir, 'logs'),
        'figures': os.path.join(experiment_dir, 'figures'),
        'checkpoints': os.path.join(experiment_dir, 'checkpoints'),
        'evaluation': os.path.join(experiment_dir, 'evaluation'),
        'explanations': os.path.join(experiment_dir, 'explanations')
    }
    
    # Create all directories
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    logging.info(f"Created experiment directory: {experiment_dir}")
    
    return subdirs


def format_metric_value(value, precision=4):
    """
    Format metric value for display.
    
    Args:
        value: Metric value
        precision: Decimal precision
        
    Returns:
        str: Formatted value
    """
    if isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        return f"{value:.{precision}f}"
    else:
        return str(value)


if __name__ == "__main__":
    # Test utilities
    set_global_seeds(42)
    setup_logging()
    
    # Get system info
    info = get_system_info()
    print("System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")