"""
Main execution script for LungConVT lung disease classification.

This script orchestrates the complete pipeline from data loading
to model training, evaluation, and explanation.

Usage:
    python main.py --mode train --config config.yaml
    python main.py --mode evaluate --model_path path/to/model.h5
    python main.py --mode explain --model_path path/to/model.h5 --image_path path/to/image.jpg

Author: [Your Name]
Date: [Current Date]
"""

import argparse
import os
import yaml
import logging
from datetime import datetime

# Import project modules
from data_loader import LungImageDataLoader, create_data_generators
from model import create_lungconvt_model
from train import ModelTrainer, calculate_class_weights
from evaluate import ModelEvaluator
from explain import ModelExplainer
from utils import (
    set_global_seeds, setup_logging, create_experiment_directory,
    save_system_info, plot_training_history, create_data_distribution_plot,
    get_model_summary, NumpyEncoder
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config):
    """
    Complete training pipeline.
    
    Args:
        config: Configuration dictionary
    """
    # Set up experiment directory
    experiment_dirs = create_experiment_directory(
        base_dir=config.get('output_dir', 'experiments'),
        experiment_name=config.get('experiment_name')
    )
    
    # Set up logging
    setup_logging(log_dir=experiment_dirs['logs'])
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training pipeline...")
    logger.info(f"Configuration: {config}")
    
    # Set random seeds
    set_global_seeds(config.get('seed', 42))
    
    # Save system info
    save_system_info(os.path.join(experiment_dirs['logs'], 'system_info.json'))
    
    # Load data
    logger.info("Loading data...")
    data_loader = LungImageDataLoader(
        csv_path=config['data']['csv_path'],
        image_size=tuple(config['model']['input_size']),
        batch_size=config['training']['batch_size'],
        seed=config.get('seed', 42)
    )
    
    data_loader.load_data()
    data_loader.split_data(test_size=config['data'].get('test_size', 0.2))
    
    # Create data distribution plot
    create_data_distribution_plot(
        data_loader,
        save_path=os.path.join(experiment_dirs['figures'], 'data_distribution.png')
    )
    
    # Create data generators
    train_generator = data_loader.get_train_generator(
        augmentation_params=config['data'].get('augmentation')
    )
    test_generator = data_loader.get_test_generator()
    
    # Create model
    logger.info("Creating model...")
    model = create_lungconvt_model(
        input_shape=tuple(config['model']['input_size']) + (3,),
        n_classes=len(data_loader.labels)
    )
    
    # Save model summary
    summary = get_model_summary(
        model,
        save_path=os.path.join(experiment_dirs['logs'], 'model_summary.txt')
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model,
        model_name=config.get('model_name', 'lungconvt'),
        output_dir=experiment_dirs['root']
    )
    
    # Compile model
    trainer.compile_model(
        learning_rate=config['training']['learning_rate'],
        optimizer_params=config['training'].get('optimizer_params')
    )
    
    # Calculate class weights if needed
    class_weights = None
    if config['training'].get('use_class_weights', False):
        class_weights = calculate_class_weights(train_generator)
    
    # Train model
    history = trainer.train(
        train_generator=train_generator,
        val_generator=test_generator,
        epochs=config['training']['epochs'],
        class_weights=class_weights
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(experiment_dirs['figures'], 'training_history.png')
    )
    
    # Load best model and evaluate
    logger.info("Evaluating model...")
    trainer.load_best_model()
    
    evaluator = ModelEvaluator(
        trainer.model,
        test_generator,
        data_loader.labels,
        output_dir=experiment_dirs['evaluation']
    )
    
    metrics = evaluator.run_full_evaluation()
    
    logger.info("Training pipeline complete!")
    
    # Save configuration
    config['results'] = {
        'experiment_dir': experiment_dirs['root'],
        'final_accuracy': metrics['accuracy'],
        'labels': data_loader.labels
    }
    
    with open(os.path.join(experiment_dirs['root'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    return experiment_dirs['root']


def evaluate_model(model_path, config):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        config: Configuration dictionary
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Evaluating model: {model_path}")
    
    # Create data generator
    _, test_generator, labels = create_data_generators(
        csv_path=config['data']['csv_path'],
        image_size=tuple(config['model']['input_size']),
        batch_size=config['training']['batch_size'],
        test_size=config['data'].get('test_size', 0.2)
    )
    
    # Load model
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    
    # Create evaluator
    output_dir = config.get('output_dir', 'outputs/evaluation')
    evaluator = ModelEvaluator(model, test_generator, labels, output_dir)
    
    # Run evaluation
    metrics = evaluator.run_full_evaluation()
    
    logger.info("Evaluation complete!")
    
    return metrics


def explain_predictions(model_path, image_paths, config):
    """
    Generate explanations for predictions.
    
    Args:
        model_path: Path to saved model
        image_paths: List of image paths to explain
        config: Configuration dictionary
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating explanations for {len(image_paths)} images")
    
    # Load model
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    
    # Get labels from config
    labels = config.get('labels', ['COVID-19', 'Normal', 
                                   'Pneumonia-Bacterial', 'Pneumonia-Viral'])
    
    # Create explainer
    output_dir = config.get('output_dir', 'outputs/explanations')
    explainer = ModelExplainer(model, labels, output_dir)
    
    # Generate explanations
    results = []
    for img_path in image_paths:
        result = explainer.visualize_grad_cam(img_path)
        results.append(result)
    
    logger.info("Explanations complete!")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='LungConVT: Lung Disease Classification'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'explain'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to saved model (for evaluate/explain modes)'
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        nargs='+',
        help='Path(s) to image(s) for explanation'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='Name for the experiment'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override experiment name if provided
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    
    # Execute based on mode
    if args.mode == 'train':
        experiment_dir = train_model(config)
        print(f"\nTraining complete! Results saved to: {experiment_dir}")
        
    elif args.mode == 'evaluate':
        if not args.model_path:
            raise ValueError("--model_path required for evaluate mode")
            
        metrics = evaluate_model(args.model_path, config)
        print(f"\nEvaluation complete! Accuracy: {metrics['accuracy']:.4f}")
        
    elif args.mode == 'explain':
        if not args.model_path:
            raise ValueError("--model_path required for explain mode")
        if not args.image_path:
            raise ValueError("--image_path required for explain mode")
            
        results = explain_predictions(args.model_path, args.image_path, config)
        print(f"\nGenerated explanations for {len(results)} images")


if __name__ == "__main__":
    main()