"""
Model explainability module using Grad-CAM and other techniques.

This module provides visualization methods to understand model predictions,
including Grad-CAM heatmaps for interpreting which image regions influence
the model's decisions.

References:
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
  via Gradient-based Localization", ICCV 2017

Author: [Your Name]
Date: [Current Date]
"""

import os
import cv2
import numpy as np

# Fix for matplotlib backend in headless environments
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import logging
from PIL import Image

# Import NumpyEncoder from utils if available
try:
    from utils import NumpyEncoder
except ImportError:
    # Define it here if utils is not available
    class NumpyEncoder(json.JSONEncoder):
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Model explanation class using various visualization techniques.
    
    Primarily implements Grad-CAM for visual explanations of CNN predictions.
    """
    
    def __init__(self, model, labels, output_dir='outputs/explanations'):
        """
        Initialize the explainer.
        
        Args:
            model: Trained Keras model
            labels: List of class labels
            output_dir: Directory to save explanations
        """
        self.model = model
        self.labels = labels
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def grad_cam(self, img_array, class_idx, layer_name=None):
        """
        Generate Grad-CAM heatmap for a specific class.
        
        Args:
            img_array: Preprocessed image array (batch_size, H, W, C)
            class_idx: Index of the target class
            layer_name: Name of convolutional layer to visualize
                       (default: last conv layer)
                       
        Returns:
            tuple: (heatmap, predictions)
        """
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Conv layer
                    layer_name = layer.name
                    break
            logger.info(f"Using layer: {layer_name}")
            
        # Get model outputs
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
            
        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Average gradients spatially
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
            
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        
        return heatmap, predictions.numpy()
        
    def visualize_grad_cam(self, img_path, save_path=None, layer_name=None, 
                          alpha=0.4, target_size=(256, 256)):
        """
        Create Grad-CAM visualization for an image.
        
        Args:
            img_path: Path to input image
            save_path: Path to save visualization
            layer_name: Target layer name
            alpha: Transparency for heatmap overlay
            target_size: Image size for model input
            
        Returns:
            dict: Visualization results
        """
        # Load and preprocess image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Get predictions
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        logger.info(f"Predicted: {self.labels[predicted_class]} "
                   f"(confidence: {confidence:.3f})")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Grad-CAM Analysis: {os.path.basename(img_path)}', 
                    fontsize=16)
        
        # Original image
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Prediction bar chart
        axes[0, 1].bar(self.labels, predictions[0])
        axes[0, 1].set_title('Prediction Probabilities')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Generate Grad-CAM for predicted class
        heatmap, _ = self.grad_cam(img_array, predicted_class, layer_name)
        
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, target_size)
        
        # Display heatmap
        axes[0, 2].imshow(heatmap_resized, cmap='jet')
        axes[0, 2].set_title(f'Grad-CAM: {self.labels[predicted_class]}')
        axes[0, 2].axis('off')
        
        # Superimposed visualization
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        superimposed = heatmap_colored * alpha + np.array(img) / 255.0 * (1 - alpha)
        
        axes[1, 0].imshow(superimposed)
        axes[1, 0].set_title('Superimposed (Predicted Class)')
        axes[1, 0].axis('off')
        
        # Generate Grad-CAM for other top classes
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        
        for idx, (ax, class_idx) in enumerate(zip(axes[1, 1:], top_indices[1:3])):
            if class_idx < len(self.labels):
                heatmap_other, _ = self.grad_cam(img_array, class_idx, layer_name)
                heatmap_other_resized = cv2.resize(heatmap_other, target_size)
                heatmap_other_colored = plt.cm.jet(heatmap_other_resized)[:, :, :3]
                superimposed_other = (heatmap_other_colored * alpha + 
                                    np.array(img) / 255.0 * (1 - alpha))
                
                ax.imshow(superimposed_other)
                ax.set_title(f'{self.labels[class_idx]} '
                           f'({predictions[0][class_idx]:.3f})')
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                f'gradcam_{os.path.basename(img_path)}'
            )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Important: close figure to free memory
        
        logger.info(f"Grad-CAM visualization saved to {save_path}")
        
        # Return results
        results = {
            'image_path': img_path,
            'predicted_class': self.labels[predicted_class],
            'confidence': float(confidence),
            'predictions': {label: float(prob) 
                          for label, prob in zip(self.labels, predictions[0])},
            'visualization_path': save_path
        }
        
        return results
        
    def analyze_misclassifications(self, test_generator, num_samples=10):
        """
        Analyze misclassified samples using Grad-CAM.
        
        Args:
            test_generator: Test data generator
            num_samples: Number of misclassified samples to analyze
        """
        logger.info("Analyzing misclassified samples...")
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.labels
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
            
        # Find misclassified indices
        misclassified_indices = np.where(y_pred != y_true)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassifications found!")
            return
            
        logger.info(f"Found {len(misclassified_indices)} misclassifications")
        
        # Create directory for misclassification analysis
        misclass_dir = os.path.join(self.output_dir, 'misclassifications')
        os.makedirs(misclass_dir, exist_ok=True)
        
        # Analyze samples
        num_to_analyze = min(num_samples, len(misclassified_indices))
        sample_indices = np.random.choice(misclassified_indices, 
                                        num_to_analyze, 
                                        replace=False)
        
        for idx in sample_indices:
            # Get image path from generator
            img_path = test_generator.filepaths[idx]
            true_label = self.labels[y_true[idx]]
            pred_label = self.labels[y_pred[idx]]
            
            logger.info(f"Analyzing: True={true_label}, Predicted={pred_label}")
            
            # Create Grad-CAM visualization
            save_name = f"misclass_{idx}_true_{true_label}_pred_{pred_label}.png"
            save_path = os.path.join(misclass_dir, save_name)
            
            self.visualize_grad_cam(img_path, save_path)
            
    def generate_layer_activations(self, img_path, layer_names=None, 
                                 target_size=(256, 256)):
        """
        Visualize intermediate layer activations.
        
        Args:
            img_path: Path to input image
            layer_names: List of layer names to visualize
            target_size: Image size for model input
        """
        # Load and preprocess image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Select layers to visualize
        if layer_names is None:
            layer_names = []
            for layer in self.model.layers:
                if 'conv' in layer.name.lower():
                    layer_names.append(layer.name)
            # Limit to first few conv layers
            layer_names = layer_names[:6]
            
        # Create activation models
        activation_models = []
        for layer_name in layer_names:
            try:
                layer = self.model.get_layer(layer_name)
                activation_model = tf.keras.models.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )
                activation_models.append((layer_name, activation_model))
            except:
                logger.warning(f"Could not get layer: {layer_name}")
                
        # Generate activations
        fig, axes = plt.subplots(
            len(activation_models), 
            8, 
            figsize=(20, 3 * len(activation_models))
        )
        
        if len(activation_models) == 1:
            axes = axes.reshape(1, -1)
            
        for row, (layer_name, activation_model) in enumerate(activation_models):
            activations = activation_model.predict(img_array)
            
            # Display first 8 filters
            for col in range(min(8, activations.shape[-1])):
                ax = axes[row, col]
                ax.imshow(activations[0, :, :, col], cmap='viridis')
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(layer_name, rotation=90, size='large')
                    
        plt.suptitle('Layer Activations', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(
            self.output_dir,
            f'activations_{os.path.basename(img_path)}'
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Important: close figure to free memory
        
        logger.info(f"Activation visualization saved to {save_path}")
        
    def batch_explain(self, image_paths, layer_name=None):
        """
        Generate explanations for multiple images.
        
        Args:
            image_paths: List of image paths
            layer_name: Target layer for Grad-CAM
            
        Returns:
            list: Results for each image
        """
        results = []
        
        for img_path in image_paths:
            logger.info(f"Processing {img_path}")
            result = self.visualize_grad_cam(img_path, layer_name=layer_name)
            results.append(result)
            
        # Save summary
        import json
        summary_path = os.path.join(self.output_dir, 'batch_explanation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
            
        logger.info(f"Batch explanation complete. Summary saved to {summary_path}")
        
        return results


def explain_predictions(model_path, test_images, labels, output_dir='outputs/explanations'):
    """
    Convenience function to explain model predictions.
    
    Args:
        model_path: Path to saved model
        test_images: List of test image paths
        labels: List of class labels
        output_dir: Output directory
    """
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create explainer
    explainer = ModelExplainer(model, labels, output_dir)
    
    # Generate explanations
    results = explainer.batch_explain(test_images)
    
    return results


if __name__ == "__main__":
    # Example usage
    labels = ['COVID-19', 'Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral']
    
    # Load model and create explainer
    model = tf.keras.models.load_model('outputs/models/lungconvt_best.h5')
    explainer = ModelExplainer(model, labels)
    
    # Example: Explain single image
    result = explainer.visualize_grad_cam('path/to/test/image.jpg')
    print(f"Prediction: {result['predicted_class']} "
          f"(confidence: {result['confidence']:.3f})")