#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Particle Swarm Optimization (PSO) Module

Implements PSO algorithm for neural network weight optimization.
"""

import numpy as np


class Particle:
    """Represents a particle in the PSO algorithm."""
    
    def __init__(self, dimensions, random_seed=None):
        """Initialize a particle with random position and velocity.
        
        Args:
            dimensions (int): Number of dimensions in search space
            random_seed (int, optional): Random seed for reproducibility
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize position and velocity randomly
        self.position = np.random.uniform(-1, 1, dimensions)
        self.velocity = np.random.uniform(-0.1, 0.1, dimensions)
        
        # Initialize personal best
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')  # For minimization problem
    
    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """Update the velocity of the particle.
        
        Args:
            global_best_position (np.ndarray): Global best position
            w (float): Inertia weight
            c1 (float): Cognitive coefficient
            c2 (float): Social coefficient
        """
        # Generate random values for cognitive and social components
        r1 = np.random.random(self.position.shape)
        r2 = np.random.random(self.position.shape)
        
        # Calculate cognitive component (personal best)
        cognitive = c1 * r1 * (self.best_position - self.position)
        
        # Calculate social component (global best)
        social = c2 * r2 * (global_best_position - self.position)
        
        # Update velocity with inertia, cognitive, and social components
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self, bounds=None):
        """Update the position of the particle.
        
        Args:
            bounds (tuple, optional): (min_bound, max_bound) for position values
        """
        # Update position with current velocity
        self.position = self.position + self.velocity
        
        # Apply bounds if provided
        if bounds is not None:
            min_bound, max_bound = bounds
            self.position = np.clip(self.position, min_bound, max_bound)
    
    def evaluate(self, fitness_function):
        """Evaluate the particle's position using the fitness function.
        
        Args:
            fitness_function (callable): Function to evaluate fitness
            
        Returns:
            float: Fitness value
        """
        # Calculate fitness
        fitness = fitness_function(self.position)
        
        # Update personal best if current position is better
        if fitness < self.best_fitness:  # For minimization problem
            self.best_fitness = fitness
            self.best_position = self.position.copy()
        
        return fitness


class PSO:
    """Particle Swarm Optimization (PSO) implementation."""
    
    def __init__(self, dimensions, num_particles=30, max_iterations=100, random_seed=42):
        """Initialize the PSO algorithm.
        
        Args:
            dimensions (int): Number of dimensions in search space
            num_particles (int): Number of particles in the swarm
            max_iterations (int): Maximum number of iterations
            random_seed (int): Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w_decay = 0.99  # Inertia weight decay
        
        # Initialize particles
        self.particles = [
            Particle(dimensions, random_seed + i) 
            for i in range(num_particles)
        ]
        
        # Initialize global best
        self.global_best_position = None
        self.global_best_fitness = float('inf')  # For minimization problem
        
        # History for tracking progress
        self.fitness_history = []
    
    def optimize(self, fitness_function, bounds=(-1, 1), verbose=False):
        """Run the PSO optimization algorithm.
        
        Args:
            fitness_function (callable): Function to evaluate fitness
            bounds (tuple): (min_bound, max_bound) for position values
            verbose (bool): Whether to print progress
            
        Returns:
            tuple: (best_position, best_fitness)
        """
        # Initialize global best from initial particle positions
        for particle in self.particles:
            fitness = particle.evaluate(fitness_function)
            
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
        
        # Main PSO loop
        for iteration in range(self.max_iterations):
            # Update inertia weight with decay
            current_w = self.w * (self.w_decay ** iteration)
            
            # Update all particles
            for particle in self.particles:
                # Update velocity and position
                particle.update_velocity(self.global_best_position, current_w, self.c1, self.c2)
                particle.update_position(bounds)
                
                # Evaluate new position
                fitness = particle.evaluate(fitness_function)
                
                # Update global best if needed
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
            
            # Record best fitness for history
            self.fitness_history.append(self.global_best_fitness)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, "
                      f"Best Fitness: {self.global_best_fitness:.6f}")
        
        if verbose:
            print(f"\nOptimization completed:")
            print(f"Best Fitness: {self.global_best_fitness:.6f}")
        
        return self.global_best_position, self.global_best_fitness 