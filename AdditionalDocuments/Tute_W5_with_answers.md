# Tutorial – Week 5

## 1.  Here is the ANN implementation of XOR source code file.

**Code of ANN implementation:**

```python
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 6 20:53:46 2021
@author: Mary Tom
"""

import numpy as np

#np.random.seed(0)

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

epochs = 10000
lr = 0.1 #learning rate

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

#Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))

#display initial values
print("Initial hidden weights: ", end='')
print(*hidden_weights)
print("Initial hidden biases: ", end='')
print(*hidden_bias)
print("Initial output weights: ", end='')
print(*output_weights)
print("Initial output biases: ", end='')
print(*output_bias)

#Training algorithm
for _ in range(epochs):
    #Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    #Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("Final hidden weights: ", end='')
print(*hidden_weights)
print("Final hidden bias: ", end='')
print(*hidden_bias)
print("Final output weights: ", end='')
print(*output_weights)
print("Final output bias: ", end='')
print(*output_bias)
print("
Output from neural network after 10,000 epochs: ", end='')
print(*predicted_output)
```

### 1.1  Run the program with different number of epochs and learning rate. Create a table and record the results obtained including the final error.

**Answer of 1.1:**

```python
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 6 20:53:46 2021
@author: Mary Tom
"""

import numpy as np

#np.random.seed(0)

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

epochs = 50
lr = 0.1 #learning rate

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

#Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))

#display initial values
print("Initial hidden weights: ", end='')
print(*hidden_weights)
print("Initial hidden biases: ", end='')
print(*hidden_bias)
print("Initial output weights: ", end='')
print(*output_weights)
print("Initial output biases: ", end='')
print(*output_bias)

#Training algorithm
for _ in range(epochs):
    #Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    #Backpropagation
    error = expected_output - predicted_output
    print(error)
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("Final hidden weights: ", end='')
print(*hidden_weights)
print("Final hidden bias: ", end='')
print(*hidden_bias)
print("Final output weights: ", end='')
print(*output_weights)
print("Final output bias: ", end='')
print(*output_bias)
print("
Output from neural network after 10,000 epochs: ", end='')
print(*predicted_output)
```

### 1.2  Create PSO class to create weights for creating and updating the weights for the ANN implementation of XOR.

**Answer of 1.2:**

```python
# -*- coding: utf-8 -*-

"""
Created on Friday Jul 10 16:55:02 2020
@author: Mary Tom
Reference: Particle Swarm Optimization - Clerk Chapter 6
"""

"""
the cost function is f(x) = for d=1 to D sum(x sin(x)+ 0.1x) where d is the dimensions.
for a two dimension, it is x1 sin(x1)+0.1x1+ x2 sin(x2)+0.1x2
Minimize f(x) where the boundaries are -2 to 2 for x.
the lower and upper boundary of x dimension is -2 and 2
new position x is calculated from the previous and new velocity
x (t+1 ) = x(t) +v(t+1)
new velocity is calculated using the following relation
v(t+1) = wv(t)+ c1r1(p(t)-x(t)) + C2r2 (G- x(t))
p(t) is the particle's best and G is the global best
"""

import random
import math
import numpy as np

SWARM_SIZE_MAX = 40
MAX_DIMENSIONS = 10

swarm_size = 5  ## N
informants = 3   # K
dimensions = 9   # D

TWO_PI = 6.283185307
E = 2.7182818285

max_boundary = 1
min_boundary = 0

c1 = 0.689343
cmax = 1.42694

best_global_positions = [1.8 for v in range(dimensions)]
best_global_fitness = 800

############################################################
class Particle:
    def __init__(self, dimensions, minValue, maxValue):
        self.dimensions = dimensions
        self.min_boundary = minValue
        self.max_boundary = maxValue
        self.positions = [1 for d in range(dimensions)]
        self.velocities = [1 for d in range(dimensions)]
        #init x and y values
        self.informant = self
        for j in range(0, dimensions):
            value = random.random()
            scaled_value = min_boundary + (value * (max_boundary - min_boundary))
            self.positions[j] = scaled_value
        #init velocities of x and y
        value = random.random()
        scaled_value = min_boundary/2 + (value * (((max_boundary - min_boundary)/2) - ((min_boundary - max_boundary)/2)))
        self.velocities[j] = scaled_value
        # maximum fitness of self
        self.fitness = 100
        self.pbest = 100
        self.best_positions = [1.0 for v in range(dimensions)]
        self.group_best = 100
        self.group_positions = [1.0 for d in range(dimensions)]

    def calculateFitness(self, error):
        average = np.average(error)
        self.fitness = average
        print("fitness")
        print(self.fitness)
        return self.fitness

    def get_positions(self):
        return self.positions

    def modifyPosition(self, new_positions):
        for i in range(0, self.dimensions):
            self.positions[i] = new_positions[i]

    def set_memory(self, memory):
        self.informant = memory

    def set_informants(self):
        # Not used in this implementation
        pass

    def set_group_best(self, swarm):
        best = 100
        best_index = -1
        best = swarm[0].pbest
        for i in range(1, swarm_size):
            if (swarm[i].pbest < best):
                best = swarm[i].pbest
                best_index = i
        self.group_best = best
        for j in range(0, dimensions):
            self.group_positions[j] = swarm[best_index].positions[j]

    def modify_group_best(self, swarm):
        best_index = -1
        for i in range(0, swarm_size):
            best = swarm[i].pbest
            if (best < self.group_best):
                self.group_best = best
                best_index = i
        for j in range(0, dimensions):
            self.group_positions[j] = swarm[best_index].positions[j]

    def calculate_new_velocity(self):
        for d in range(dimensions):
            part2 = random.uniform(0, cmax) * (self.best_positions[d] - self.positions[d])
            self.velocities[d] = c1 * self.velocities[d] + part2
            part3 = random.uniform(0, cmax) * (self.informant.group_positions[d] - self.positions[d])
            self.velocities[d] = self.velocities[d] + part3

    def displayParticle(self):
        print("Particle details:")
        for i in range(0, self.dimensions):
            print("Positions:" + str(self.positions[i]) + " ")
            print("velocities: " + str(self.velocities[i]) + " ")
        print("fitness: " + str(self.fitness) + " pbest: " + str(self.pbest) + " group best : " + str(self.group_best))
```

### 1.3  Use the PSO class methods in the ANN implementation of XOR to update weights replacing the backpropagation currently used.

**Answer of 1.3:**

```python
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 6 20:53:46 2021
@author: Mary Tom
"""

import numpy as np
from pso_xor_nn import Particle
from pso_xor_nn import swarm_size
#np.random.seed(0)
informants = 3
dimensions = 9
TWO_PI = 6.283185307
E = 2.7182818285
max_boundary = 1
min_boundary = 0
c1 = 0.689343
cmax = 1.42694
desired_precision = 0.001
fmin = 0
w = 0.7

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

epochs = 5
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

memory = Particle(dimensions, min_boundary, max_boundary)
swarm = [Particle(dimensions, min_boundary, max_boundary) for x in range(swarm_size)]

for particle in swarm:
    for d in range(particle.dimensions):
        particle.best_positions[d] = particle.positions[d]
    particle.set_memory(memory)
    particle.informant.set_group_best(swarm)
    memory.set_group_best(swarm)

hidden_weights = np.empty((inputLayerNeurons, hiddenLayerNeurons), dtype=float)
hidden_bias = np.empty((hiddenLayerNeurons), dtype=float)
output_weights = np.empty((hiddenLayerNeurons, outputLayerNeurons), dtype=float)
output_bias = np.empty((outputLayerNeurons), dtype=float)

c = 0
for d in range(inputLayerNeurons):
    for j in range(hiddenLayerNeurons):
        hidden_weights[d][j] = memory.get_positions()[c]
        c = c + 1

c = 4
for d in range(hiddenLayerNeurons):
    hidden_bias[d] = memory.get_positions()[c]
    c = c + 1

for j in range(hiddenLayerNeurons):
    for k in range(outputLayerNeurons):
        output_weights[j][k] = memory.get_positions()[c]
        c = c + 1

for d in range(outputLayerNeurons):
    output_bias[d] = memory.get_positions()[c]
    c = c + 1

print("Initial hidden weights: ", end='')
print(*hidden_weights)
print("Initial hidden biases: ", end='')
print(*hidden_bias)
print("Initial output weights: ", end='')
print(*output_weights)
print("Initial output biases: ", end='')
print(*output_bias)

for _ in range(epochs):
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    print(predicted_output)
    print("expected")
    
    error = expected_output - predicted_output
    print(error)
    
    memory.calculateFitness(error)
    memory.calculate_new_velocity()
    
    for d in range(dimensions):
        memory.positions[d] += memory.velocities[d]
    memory.modify_group_best(swarm)
    
    for particle in swarm:
        particle.informant.modify_group_best(swarm)
        particle.calculateFitness(error)
        particle.calculate_new_velocity()
        for d in range(dimensions):
            if particle.velocities[d] < min_boundary:
                particle.velocities[d] = min_boundary
            elif particle.velocities[d] > max_boundary:
                particle.velocities[d] = max_boundary
        for d in range(dimensions):
            particle.positions[d] += particle.velocities[d]
            if particle.positions[d] < min_boundary:
                particle.positions[d] = min_boundary
            elif particle.positions[d] > max_boundary:
                particle.positions[d] = max_boundary
        if particle.fitness < particle.pbest:
            particle.pbest = particle.fitness
            particle.best_positions = list(particle.positions)
    
    c = 0
    for d in range(inputLayerNeurons):
        for j in range(hiddenLayerNeurons):
            hidden_weights[d][j] = memory.get_positions()[c]
            c = c + 1
    c = 4
    for d in range(hiddenLayerNeurons):
        hidden_bias[d] = memory.get_positions()[c]
        c = c + 1
    for j in range(hiddenLayerNeurons):
        for k in range(outputLayerNeurons):
            output_weights[j][k] = memory.get_positions()[c]
            c = c + 1
    for d in range(outputLayerNeurons):
        output_bias[d] = memory.get_positions()[c]
        c = c + 1

    print("Final hidden weights: ", end='')
    print(*hidden_weights)
    print("Final hidden bias: ", end='')
    print(*hidden_bias)
    print("Final output weights: ", end='')
    print(*output_weights)
    print("Final output bias: ", end='')
    print(*output_bias)
    print("
Output from neural network after 10,000 epochs: ", end='')
    print(*predicted_output)
```