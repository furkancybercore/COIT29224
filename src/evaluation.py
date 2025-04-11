#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation Module for Neural Network with PSO

Provides metrics calculation and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error
)


class Evaluator:
    """Evaluation tools for neural network performance."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def classification_metrics(self, y_true, y_pred):
        """Calculate classification metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            dict: Dictionary with various metrics
        """
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Try to calculate additional metrics (may fail for binary data)
        try:
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
        except:
            precision = recall = f1 = np.nan
        
        # Return metrics as dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            dict: Dictionary with various metrics
        """
        # Calculate MSE
        mse = mean_squared_error(y_true, y_pred)
        
        # Calculate RMSE
        rmse = np.sqrt(mse)
        
        # Calculate MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate R²
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Return metrics as dictionary
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (list, optional): Names of classes
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Set labels
        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history.
        
        Args:
            history (list or dict): Training history with metrics
        """
        # Create figure
        plt.figure(figsize=(10, 4))
        
        # If history is a list (simple fitness values)
        if isinstance(history, list):
            plt.subplot(1, 1, 1)
            plt.plot(history)
            plt.title('Training Progress')
            plt.xlabel('Iteration')
            plt.ylabel('Fitness')
            plt.grid(True)
        
        # If history is a dictionary (with multiple metrics)
        elif isinstance(history, dict):
            # Plot loss
            if 'loss' in history:
                plt.subplot(1, 2, 1)
                plt.plot(history['loss'], label='Train')
                if 'val_loss' in history:
                    plt.plot(history['val_loss'], label='Validation')
                plt.title('Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            
            # Plot accuracy
            if 'accuracy' in history:
                plt.subplot(1, 2, 2)
                plt.plot(history['accuracy'], label='Train')
                if 'val_accuracy' in history:
                    plt.plot(history['val_accuracy'], label='Validation')
                plt.title('Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_regression_results(self, y_true, y_pred):
        """Plot regression results.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        """
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of true vs predicted
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, metrics):
        """Print summary of metrics.
        
        Args:
            metrics (dict): Dictionary with metrics
        """
        print("\n" + "="*50)
        print(" Performance Metrics ".center(50, "="))
        print("="*50)
        
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        print("="*50) 