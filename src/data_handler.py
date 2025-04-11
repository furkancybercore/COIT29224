#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Handler Module for Neural Network with PSO

Handles data loading, preprocessing, and splitting.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataHandler:
    """Data handling class for loading and preprocessing datasets."""
    
    def __init__(self, random_seed=42):
        """Initialize the data handler.
        
        Args:
            random_seed (int): Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_csv(self, filepath, target_column=-1, header=True):
        """Load data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            target_column (int): Index of the target column (-1 for last column)
            header (bool): Whether the CSV has a header row
            
        Returns:
            tuple: (features, targets)
        """
        # Placeholder implementation
        print(f"Loading data from {filepath}...")
        print("Note: This is a placeholder implementation.")
        
        # In a real implementation, we would use pandas or numpy to load the data
        
        return np.array([]), np.array([])
    
    def generate_demo_data(self, n_samples=100, n_features=2, n_classes=2):
        """Generate synthetic data for demonstration.
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features
            n_classes (int): Number of classes (for classification)
            
        Returns:
            tuple: (features, targets)
        """
        # Placeholder implementation
        print(f"Generating demo data with {n_samples} samples...")
        print("Note: This is a placeholder implementation.")
        
        # Generate random data - this would be replaced with sklearn's make_classification
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, size=n_samples)
        
        return X, y
    
    def preprocess(self, X, y, test_size=0.2):
        """Preprocess and split the data.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed
        )
        
        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Store the data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test 