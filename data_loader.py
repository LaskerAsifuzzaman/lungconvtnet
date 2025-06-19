"""
Data loading and preprocessing module for lung image classification.

This module handles data loading from CSV files, image preprocessing,
and data augmentation for training and validation datasets.

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import logging

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LungImageDataLoader:
    """
    Data loader for lung X-ray image classification.
    
    Handles loading of image data from CSV files and creates
    data generators for model training and evaluation.
    
    Attributes:
        csv_path (str): Path to the CSV file containing image metadata
        image_size (tuple): Target size for resizing images (height, width)
        batch_size (int): Batch size for data generators
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, csv_path, image_size=(256, 256), batch_size=16, seed=42):
        """
        Initialize the data loader.
        
        Args:
            csv_path (str): Path to CSV file with image metadata
            image_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for generators
            seed (int): Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.labels = None
        self.train_df = None
        self.test_df = None
        
    def load_data(self):
        """Load and preprocess the CSV data."""
        logger.info(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        
        # Get unique labels
        self.labels = self.df['Folder Name'].unique().tolist()
        logger.info(f"Found {len(self.labels)} classes: {self.labels}")
        
        # Create one-hot encoded columns
        self._create_one_hot_encoding()
        
        return self.df
    
    def _create_one_hot_encoding(self):
        """Create one-hot encoded columns for each label."""
        # Create a new dataframe with one-hot encoding
        new_df = pd.DataFrame()
        
        # Add one-hot encoded columns
        for label in self.labels:
            new_df[label] = self.df['Folder Name'].apply(lambda x: 1 if x == label else 0)
        
        # Add original columns
        new_df['Folder Name'] = self.df['Folder Name']
        new_df['File Name'] = self.df['File Name']
        new_df['Complete Path'] = self.df['Complete Path']
        
        # Reorder columns
        cols = ['Folder Name', 'File Name'] + self.labels + ['Complete Path']
        self.df = new_df[cols]
        
    def split_data(self, test_size=0.2, random_state=None):
        """
        Split data into training and test sets.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (train_df, test_df)
        """
        if random_state is None:
            random_state = self.seed
            
        self.train_df, self.test_df = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.df['Folder Name']
        )
        
        logger.info(f"Training samples: {len(self.train_df)}")
        logger.info(f"Test samples: {len(self.test_df)}")
        
        return self.train_df, self.test_df
    
    def get_train_generator(self, augmentation_params=None):
        """
        Create training data generator with augmentation.
        
        Args:
            augmentation_params (dict): Parameters for data augmentation
            
        Returns:
            ImageDataGenerator: Generator for training data
        """
        if augmentation_params is None:
            augmentation_params = {
                'shear_range': 0.1,
                'zoom_range': 0.15,
                'rotation_range': 5,
                'width_shift_range': 0.1,
                'height_shift_range': 0.05,
                'rescale': 1.0/255.0
            }
        
        logger.info("Creating training data generator with augmentation")
        
        train_datagen = ImageDataGenerator(**augmentation_params)
        
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=None,
            x_col='Complete Path',
            y_col=self.labels,
            class_mode='raw',
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed,
            target_size=self.image_size
        )
        
        return train_generator
    
    def get_test_generator(self):
        """
        Create test data generator without augmentation.
        
        Returns:
            ImageDataGenerator: Generator for test data
        """
        logger.info("Creating test data generator")
        
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=self.test_df,
            directory=None,
            x_col='Complete Path',
            y_col=self.labels,
            class_mode='raw',
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seed,
            target_size=self.image_size
        )
        
        return test_generator
    
    def get_class_distribution(self, df=None):
        """
        Get class distribution statistics.
        
        Args:
            df (DataFrame): DataFrame to analyze (default: full dataset)
            
        Returns:
            dict: Class distribution statistics
        """
        if df is None:
            df = self.df
            
        distribution = df['Folder Name'].value_counts().to_dict()
        return distribution
    
    def display_sample_image(self, index=0):
        """
        Display a sample image from the dataset.
        
        Args:
            index (int): Index of the image to display
            
        Returns:
            PIL.Image: The loaded image
        """
        img_path = self.df['Complete Path'].iloc[index]
        img = Image.open(img_path)
        logger.info(f"Sample image shape: {img.size}")
        logger.info(f"Sample image label: {self.df['Folder Name'].iloc[index]}")
        return img


def create_data_generators(csv_path, image_size=(256, 256), batch_size=16, 
                          test_size=0.2, seed=42):
    """
    Convenience function to create train and test generators.
    
    Args:
        csv_path (str): Path to CSV file
        image_size (tuple): Target image size
        batch_size (int): Batch size
        test_size (float): Test set proportion
        seed (int): Random seed
        
    Returns:
        tuple: (train_generator, test_generator, labels)
    """
    # Initialize data loader
    data_loader = LungImageDataLoader(
        csv_path=csv_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed
    )
    
    # Load and split data
    data_loader.load_data()
    data_loader.split_data(test_size=test_size)
    
    # Create generators
    train_generator = data_loader.get_train_generator()
    test_generator = data_loader.get_test_generator()
    
    return train_generator, test_generator, data_loader.labels


if __name__ == "__main__":
    # Example usage
    train_gen, test_gen, labels = create_data_generators(
        csv_path='DB_modified_ImgDB.csv',
        image_size=(256, 256),
        batch_size=16
    )
    print(f"Labels: {labels}")
    print(f"Training batches: {len(train_gen)}")
    print(f"Test batches: {len(test_gen)}")