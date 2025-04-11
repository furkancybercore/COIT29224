#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic tests for Neural Network with PSO

Simple tests to verify functionality.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_handler import DataHandler
from src.neural_network import NeuralNetwork
from src.pso import PSO, Particle


class TestDataHandler(unittest.TestCase):
    """Tests for DataHandler class."""
    
    def test_init(self):
        """Test initialization of DataHandler."""
        handler = DataHandler(random_seed=42)
        self.assertEqual(handler.random_seed, 42)
    
    def test_generate_demo_data(self):
        """Test demo data generation."""
        handler = DataHandler(random_seed=42)
        X, y = handler.generate_demo_data(n_samples=10, n_features=2, n_classes=2)
        
        self.assertEqual(X.shape, (10, 2))
        self.assertEqual(y.shape, (10,))
        self.assertTrue(np.all(y >= 0) and np.all(y < 2))


class TestNeuralNetwork(unittest.TestCase):
    """Tests for NeuralNetwork class."""
    
    def test_init(self):
        """Test initialization of NeuralNetwork."""
        nn = NeuralNetwork(input_size=2, hidden_layers=[4, 4], output_size=1, random_seed=42)
        
        self.assertEqual(nn.input_size, 2)
        self.assertEqual(nn.hidden_layers, [4, 4])
        self.assertEqual(nn.output_size, 1)
        
        # Check layers sizes
        self.assertEqual(nn.layers_sizes, [2, 4, 4, 1])
        
        # Check weights and biases
        self.assertEqual(len(nn.weights), 3)
        self.assertEqual(len(nn.biases), 3)
        
        # Check shapes
        self.assertEqual(nn.weights[0].shape, (2, 4))
        self.assertEqual(nn.weights[1].shape, (4, 4))
        self.assertEqual(nn.weights[2].shape, (4, 1))
    
    def test_weights_vector(self):
        """Test conversion between weight matrices and vector."""
        nn = NeuralNetwork(input_size=2, hidden_layers=[3], output_size=1, random_seed=42)
        
        # Get weights as vector
        weights_vector = nn.get_weights_as_vector()
        
        # Calculate expected size: (2×3) + (3×1) + (1×3) + (1×1) = 6 + 3 + 3 + 1 = 13
        expected_size = 2*3 + 3*1 + 1*3 + 1*1
        self.assertEqual(weights_vector.shape, (expected_size,))
        
        # Modify vector
        modified_vector = weights_vector.copy()
        modified_vector[0] = 99.0
        
        # Set weights from vector and check if it worked
        nn.set_weights_from_vector(modified_vector)
        new_vector = nn.get_weights_as_vector()
        
        # Check if the modification was applied
        self.assertEqual(new_vector[0], 99.0)


class TestPSO(unittest.TestCase):
    """Tests for PSO class."""
    
    def test_particle_init(self):
        """Test particle initialization."""
        particle = Particle(dimensions=5, random_seed=42)
        
        self.assertEqual(particle.position.shape, (5,))
        self.assertEqual(particle.velocity.shape, (5,))
        
        # Best position should be a copy of position
        self.assertTrue(np.array_equal(particle.position, particle.best_position))
    
    def test_pso_init(self):
        """Test PSO initialization."""
        pso = PSO(dimensions=10, num_particles=5, max_iterations=50, random_seed=42)
        
        self.assertEqual(pso.dimensions, 10)
        self.assertEqual(pso.num_particles, 5)
        self.assertEqual(pso.max_iterations, 50)
        self.assertEqual(len(pso.particles), 5)
    
    def test_simple_optimization(self):
        """Test optimization with a simple function."""
        # Define a simple fitness function (sphere function)
        def sphere(x):
            return np.sum(x**2)
        
        # Initialize PSO with small dimensions for testing
        pso = PSO(dimensions=2, num_particles=5, max_iterations=10, random_seed=42)
        
        # Run optimization
        best_position, best_fitness = pso.optimize(sphere, verbose=False)
        
        # Check results
        self.assertEqual(best_position.shape, (2,))
        self.assertLessEqual(best_fitness, 1.0)  # Should be close to 0


if __name__ == '__main__':
    unittest.main() 