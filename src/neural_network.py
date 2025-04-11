#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural Network Module for PSO Optimization

Implements a neural network architecture with configurable layers.
"""

import numpy as np


class NeuralNetwork:
    """Neural Network implementation with configurable architecture."""
    
    def __init__(self, input_size, hidden_layers, output_size, random_seed=42):
        """Initialize the neural network.
        
        Args:
            input_size (int): Number of input features
            hidden_layers (list): List of neurons in each hidden layer
            output_size (int): Number of output neurons
            random_seed (int): Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Initialize the network architecture
        self.layers_sizes = [input_size] + hidden_layers + [output_size]
        
        # Placeholder for weights and biases
        self.weights = []
        self.biases = []
        
        # Initialize random weights and biases
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize random weights and biases."""
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layers_sizes) - 1):
            # Initialize weights with small random values
            w = np.random.randn(self.layers_sizes[i], self.layers_sizes[i+1]) * 0.1
            b = np.zeros((1, self.layers_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _sigmoid(self, x):
        """Sigmoid activation function.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Output array after sigmoid activation
        """
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid function.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Derivative of sigmoid at x
        """
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation through the network.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size)
            
        Returns:
            np.ndarray: Output predictions
        """
        # Placeholder implementation
        print("Forward propagation - placeholder implementation")
        
        activations = [X]
        for i in range(len(self.weights)):
            # Calculate the weighted sum
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            # Apply activation function
            a = self._sigmoid(z)
            activations.append(a)
        
        return activations[-1]
    
    def predict(self, X):
        """Make predictions with the network.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size)
            
        Returns:
            np.ndarray: Predicted classes or values
        """
        # Get the output from forward propagation
        output = self.forward(X)
        
        # For classification, return the class with highest probability
        if self.output_size > 1:
            return np.argmax(output, axis=1)
        
        # For regression, return the raw outputs
        return output
    
    def get_weights_as_vector(self):
        """Flatten all weights and biases into a single vector.
        
        Returns:
            np.ndarray: Flattened vector of all weights and biases
        """
        weights_vector = []
        
        # Flatten all weights
        for w in self.weights:
            weights_vector.extend(w.flatten())
        
        # Flatten all biases
        for b in self.biases:
            weights_vector.extend(b.flatten())
        
        return np.array(weights_vector)
    
    def set_weights_from_vector(self, weights_vector):
        """Set weights and biases from a flattened vector.
        
        Args:
            weights_vector (np.ndarray): Flattened vector of all weights and biases
        """
        index = 0
        
        # Set weights
        for i in range(len(self.weights)):
            shape = self.weights[i].shape
            size = shape[0] * shape[1]
            
            self.weights[i] = weights_vector[index:index+size].reshape(shape)
            index += size
        
        # Set biases
        for i in range(len(self.biases)):
            shape = self.biases[i].shape
            size = shape[0] * shape[1]
            
            self.biases[i] = weights_vector[index:index+size].reshape(shape)
            index += size 