#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural Network with PSO Optimization
Main script with CLI interface

Author: COIT29224 Student
Date: November 2025
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_handler import DataHandler
from neural_network import NeuralNetwork
from pso import PSO
from evaluation import Evaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Neural Network with PSO Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--dataset", type=str, required=False,
                        help="Path to the dataset file (CSV)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--hidden-layers", type=str, default="10,10",
                        help="Comma-separated list of neurons in hidden layers")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="Learning rate for training")
    parser.add_argument("--particles", type=int, default=30,
                        help="Number of particles for PSO")
    parser.add_argument("--output", type=str, default="models/model.pkl",
                        help="Path to save the trained model")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--load", type=str, 
                        help="Load a previously trained model")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic demo data")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Print welcome message
    print("\n" + "="*70)
    print(" Neural Network with PSO Optimization ".center(70, "="))
    print("="*70 + "\n")
    
    # Parse hidden layers configuration
    hidden_layers = [int(neurons) for neurons in args.hidden_layers.split(",")]
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  - Dataset: {args.dataset}")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Hidden Layers: {hidden_layers}")
        print(f"  - Learning Rate: {args.learning_rate}")
        print(f"  - Particles: {args.particles}")
        print(f"  - Output: {args.output}")
        print(f"  - Random Seed: {args.random_seed}")
        print()
    
    # Initialize components
    data_handler = DataHandler(random_seed=args.random_seed)
    evaluator = Evaluator()
    
    # Load or generate data
    if args.dataset:
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset file '{args.dataset}' not found.")
            sys.exit(1)
        
        print(f"Loading dataset from {args.dataset}...")
        # X, y = data_handler.load_csv(args.dataset)
    elif args.demo:
        print("Generating demo classification data...")
        X, y = data_handler.generate_demo_data(n_samples=100, n_features=5, n_classes=2)
    else:
        print("No dataset provided. To run with demo data, use --demo")
        print("Generating a small demo dataset for demonstration...")
        X, y = data_handler.generate_demo_data(n_samples=20, n_features=3, n_classes=2)
    
    # Preprocess and split data
    X_train, X_test, y_train, y_test = data_handler.preprocess(X, y, test_size=0.2)
    
    print(f"\nData split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Neural Network configuration
    input_size = X_train.shape[1]
    if len(np.unique(y)) <= 5:  # Classification
        output_size = len(np.unique(y))
        task_type = "classification"
    else:  # Regression
        output_size = 1
        task_type = "regression"
    
    print(f"\nTask type: {task_type.upper()}")
    print(f"Input size: {input_size}, Output size: {output_size}")
    
    # Initialize neural network
    nn = NeuralNetwork(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        random_seed=args.random_seed
    )
    
    # For fitness calculation, we need a function that takes weights as input
    # and returns the error/loss as output
    def fitness_function(weights):
        """Fitness function for PSO (lower is better)."""
        # Set neural network weights from the particle position
        nn.set_weights_from_vector(weights)
        
        # Forward pass to get predictions
        predictions = nn.forward(X_train)
        
        # Calculate mean squared error for both classification and regression
        if task_type == "classification":
            # One-hot encode actual labels for comparison
            y_one_hot = np.zeros((len(y_train), output_size))
            for i, label in enumerate(y_train):
                y_one_hot[i, int(label)] = 1
            
            # Mean squared error
            mse = np.mean((predictions - y_one_hot) ** 2)
        else:
            # Regression: compare directly
            mse = np.mean((predictions - y_train.reshape(-1, 1)) ** 2)
        
        return mse
    
    # Get the number of dimensions (weights) in the neural network
    weights_vector = nn.get_weights_as_vector()
    dimensions = len(weights_vector)
    
    print(f"\nNeural network has {dimensions} trainable parameters")
    
    # Initialize PSO
    pso = PSO(
        dimensions=dimensions,
        num_particles=args.particles,
        max_iterations=args.epochs,
        random_seed=args.random_seed
    )
    
    print("\nTraining neural network with PSO...")
    print("Note: This is a placeholder implementation. No actual training is performed.")
    
    # Run optimization (commented out for placeholder)
    # best_position, best_fitness = pso.optimize(
    #     fitness_function=fitness_function,
    #     bounds=(-1, 1),
    #     verbose=args.verbose
    # )
    
    # # Set the best weights to the neural network
    # nn.set_weights_from_vector(best_position)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    print("Note: Using random predictions for demonstration.")
    
    # Generate random predictions for demonstration
    if task_type == "classification":
        y_pred = np.random.randint(0, output_size, size=len(y_test))
        metrics = evaluator.classification_metrics(y_test, y_pred)
    else:
        y_pred = np.random.randn(len(y_test))
        metrics = evaluator.regression_metrics(y_test, y_pred)
    
    # Print evaluation results
    evaluator.print_summary(metrics)
    
    # Save model (placeholder)
    if args.output:
        print(f"\nSaving model to {args.output}...")
        print("Note: Model saving is not implemented in this placeholder.")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("\nImplementation steps for future development:")
    print("1. Complete data loading and preprocessing functionality")
    print("2. Finalize neural network forward propagation")
    print("3. Optimize PSO algorithm parameters")
    print("4. Implement model saving and loading")
    print("5. Add visualization of training progress")
    
    print("\n" + "="*70)
    print(" End of Execution ".center(70, "="))
    print("="*70 + "\n")


if __name__ == "__main__":
    main() 