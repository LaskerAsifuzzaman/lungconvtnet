"""
Model evaluation module for lung disease classification.

This module provides comprehensive evaluation metrics including
confusion matrix, classification report, ROC curves, and more.

Author: [Your Name]
Date: [Current Date]
"""

import os
import json
import numpy as np

# Fix for matplotlib backend in headless environments
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score
)
import tensorflow as tf
import logging

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


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Provides various evaluation metrics and visualizations
    for multi-class classification models.
    """
    
    def __init__(self, model, test_generator, labels, output_dir='outputs/evaluation'):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained Keras model
            test_generator: Test data generator
            labels: List of class labels
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.test_generator = test_generator
        self.labels = labels
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store predictions and true labels
        self.y_pred_probs = None
        self.y_pred = None
        self.y_true = None
        
    def predict(self):
        """Generate predictions on test set."""
        logger.info("Generating predictions on test set...")
        
        # Get predictions
        self.y_pred_probs = self.model.predict(
            self.test_generator,
            steps=len(self.test_generator),
            verbose=1
        )
        
        # Convert probabilities to class predictions
        self.y_pred = np.argmax(self.y_pred_probs, axis=1)
        
        # Get true labels
        self.y_true = self.test_generator.labels
        if len(self.y_true.shape) > 1:  # One-hot encoded
            self.y_true = np.argmax(self.y_true, axis=1)
            
        logger.info(f"Predictions generated for {len(self.y_pred)} samples")
        
    def evaluate_performance(self):
        """
        Evaluate model performance and save results.
        
        Returns:
            dict: Evaluation metrics
        """
        if self.y_pred is None:
            self.predict()
            
        # Calculate metrics
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = np.mean(self.y_pred == self.y_true)
        
        # Per-class metrics
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.labels,
            output_dict=True
        )
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # F1 scores
        metrics['macro_f1'] = f1_score(self.y_true, self.y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(self.y_true, self.y_pred, average='weighted')
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
            
        logger.info(f"Evaluation metrics saved to {metrics_path}")
        
        # Print summary
        self._print_evaluation_summary(metrics)
        
        return metrics
        
    def _print_evaluation_summary(self, metrics):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        print("\nPer-Class Performance:")
        print("-"*40)
        
        for label in self.labels:
            class_metrics = metrics['classification_report'][label]
            print(f"{label:20s} | "
                  f"Precision: {class_metrics['precision']:.3f} | "
                  f"Recall: {class_metrics['recall']:.3f} | "
                  f"F1: {class_metrics['f1-score']:.3f}")
        print("="*50 + "\n")
        
    def plot_confusion_matrix(self, normalize=False, figsize=(10, 8)):
        """
        Plot and save confusion matrix.
        
        Args:
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
        """
        if self.y_pred is None:
            self.predict()
            
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.labels,
            yticklabels=self.labels,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(
            self.output_dir, 
            f"confusion_matrix{'_normalized' if normalize else ''}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Important: close figure to free memory
        
        logger.info(f"Confusion matrix saved to {save_path}")
        
    def plot_roc_curves(self, figsize=(10, 8)):
        """
        Plot ROC curves for each class.
        
        Args:
            figsize: Figure size
        """
        if self.y_pred_probs is None:
            self.predict()
            
        plt.figure(figsize=figsize)
        
        # Calculate and plot ROC curve for each class
        auc_scores = []
        for i, label in enumerate(self.labels):
            # Get binary labels for current class
            y_true_binary = (self.y_true == i).astype(int)
            y_score = self.y_pred_probs[:, i]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            auc = roc_auc_score(y_true_binary, y_score)
            auc_scores.append(auc)
            
            # Plot
            plt.plot(
                fpr, tpr,
                label=f'{label} (AUC = {auc:.3f})',
                linewidth=2
            )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        
        # Formatting
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curves', fontsize=16, pad=20)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Important: close figure to free memory
        
        # Save AUC scores
        auc_dict = dict(zip(self.labels, auc_scores))
        auc_dict['average'] = np.mean(auc_scores)
        
        auc_path = os.path.join(self.output_dir, 'auc_scores.json')
        with open(auc_path, 'w') as f:
            json.dump(auc_dict, f, indent=4, cls=NumpyEncoder)
            
        logger.info(f"ROC curves saved to {save_path}")
        logger.info(f"Average AUC: {auc_dict['average']:.4f}")
        
    def plot_precision_recall_curves(self, figsize=(10, 8)):
        """
        Plot precision-recall curves for each class.
        
        Args:
            figsize: Figure size
        """
        if self.y_pred_probs is None:
            self.predict()
            
        plt.figure(figsize=figsize)
        
        for i, label in enumerate(self.labels):
            # Get binary labels for current class
            y_true_binary = (self.y_true == i).astype(int)
            y_score = self.y_pred_probs[:, i]
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            
            # Plot
            plt.plot(recall, precision, label=label, linewidth=2)
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves', fontsize=16, pad=20)
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'precision_recall_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Important: close figure to free memory
        
        logger.info(f"Precision-recall curves saved to {save_path}")
        
    def generate_classification_report(self):
        """Generate and save detailed classification report."""
        if self.y_pred is None:
            self.predict()
            
        # Generate report
        report = classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.labels,
            digits=4
        )
        
        # Save report
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(report)
            
        logger.info(f"Classification report saved to {report_path}")
        
        # Print report
        print("\nClassification Report:")
        print(report)
        
    def save_predictions(self):
        """Save predictions for further analysis."""
        if self.y_pred is None:
            self.predict()
            
        predictions_data = {
            'true_labels': self.y_true.tolist(),
            'predicted_labels': self.y_pred.tolist(),
            'prediction_probabilities': self.y_pred_probs.tolist(),
            'label_names': self.labels
        }
        
        predictions_path = os.path.join(self.output_dir, 'predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, cls=NumpyEncoder)
            
        logger.info(f"Predictions saved to {predictions_path}")
        
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        logger.info("Running full evaluation pipeline...")
        
        # Generate predictions
        self.predict()
        
        # Calculate metrics
        metrics = self.evaluate_performance()
        
        # Generate visualizations
        self.plot_confusion_matrix(normalize=False)
        self.plot_confusion_matrix(normalize=True)
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        
        # Generate reports
        self.generate_classification_report()
        self.save_predictions()
        
        logger.info("Evaluation complete!")
        
        return metrics


def evaluate_model(model_path, test_generator, labels, output_dir='outputs/evaluation'):
    """
    Convenience function to evaluate a saved model.
    
    Args:
        model_path: Path to saved model
        test_generator: Test data generator
        labels: List of class labels
        output_dir: Output directory
        
    Returns:
        dict: Evaluation metrics
    """
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_generator, labels, output_dir)
    
    # Run evaluation
    metrics = evaluator.run_full_evaluation()
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from data_loader import create_data_generators
    
    # Create test generator
    _, test_gen, labels = create_data_generators(
        csv_path='DB_modified_ImgDB.csv',
        image_size=(256, 256),
        batch_size=16
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model_path='outputs/models/lungconvt_best.h5',
        test_generator=test_gen,
        labels=labels
    )