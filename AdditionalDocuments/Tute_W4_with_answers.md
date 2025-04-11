# Tutorial – Week 4

**Q1)** The PSO optimization of Alpine function, _f(x) = x sin(x) + 0.1x_  
solution from Week 3 has introduced a memory swarm of one particle.

a.  Modify the program to increase the size of memory swarm to two  
    particles, and then to a ring of 10 particles. Use the explorer  
    swarm size also 10.

**Answer Q1a:**

```python
# -*- coding: utf-8 -*-

"""
Created on Friday Jul 10 16:55:02 2020
Modified on Wednesday March 31 2021
@author: Mary Tom
Reference: Particle Swarm Optimization - Clerk Chapter 7
"""

"""
the cost function is the Alpine function

Alpine 10D: ∑(𝑑=1^𝐷▒| X_𝑑 sin( 𝑥_𝑑)+0.1 X_d |

f(x) = for d=1 to D sum abs((x sin(x)+ 0.1x)) where d is the
dimension.

for a two dimension, it is x1 sin(x1)+0.1x1+ x2 sin(x2)+0.1x2

Minimize f(x) where the boundaries are -2 to 2 for x.

the lower and upper boundary of x is -2 and 2

new position x is calculated from the previous and new velocity

x (t+1 ) = x(t) + v(t+1)

new velocity is calculated using the following relation

v(t+1) = wv(t) + c1*r1*(p(t)-x(t)) + C2*r2*(G- x(t))

p(t) is the particle's best and G is the global best

This program uses two memory particles.
Each memory is set to have one half of the informants which are
pratciles of
explorer swarms
Each particle of explorer swarm is informed by the corresponding
memory.
The two memeoreis are not communicating to each other.
This can be observed in the results as each one has different
group_best.
An improvement can be comparing the group_best of memories and
updating the
global which may be suitable for some problems.
This is found to be the most effective for Alpine function.
"""

import random
import math
#import numpy as np

SWARM_SIZE_MAX = 40
MAX_DIMENSIONS = 10

swarm_size = 8  ## N
informants = 4   # K
dimensions = 2   # D
memory_size = 2

#LINKS list of informants this is not used
links = [[0 for v in range(swarm_size)] for v in range(informants)]

TWO_PI = 6.283185307  # not used
E = 2.718281828     # not used

max_boundary = 2
min_boundary = -2

number_of_iterations = 10
c1 = 0.689343
cmax = 1.42694
desired_precision = 0.00001
fmin = 0   # fitness or objective to reach
w = 0.7    # inertia

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
        # init x and y values
        self.informantIndices = [0 for i in range(informants)]
        self.memoryIndices = [0 for i in range(memory_size)]
        # use one memory-swarm to inform following star topology
        for j in range(0, dimensions):
            value = random.random()
            scaled_value = min_boundary + (value * (max_boundary - min_boundary))
            self.positions[j] = scaled_value
        # init velocities of x and y
        value = random.random()
        scaled_value = min_boundary/2 + (value * (((max_boundary - min_boundary)/2) - ((min_boundary - max_boundary)/2)))
        self.velocities[j] = scaled_value
        # maximum fitness of self
        self.fitness = 100
        self.pbest = 100
        self.best_positions = [1.0 for v in range(dimensions)]
        self.group_best = 100   # this value is only used for the memory particle
        self.group_positions = [1.0 for d in range(dimensions)]

    def calculateFitness(self):
        f = 0
        for i in range(0, self.dimensions):
            f = f + abs(self.positions[i] * math.sin(self.positions[i]) + 0.1 * self.positions[i])
        self.fitness = f
        return f

    def modifyPosition(self, new_positions):
        for i in range(0, self.dimensions):
            self.positions[i] = new_positions[i]

    # this is method for setting memories for the explorer swarm
    def set_memories(self, memoryIndex):
        self.memoryIndices[0] = memoryIndex
        # print(self.memoryIndices)

    # this method should be used by memory swarm
    def set_informants(self, memoryIndex):
        col = 0
        if (memoryIndex == 0):
            for j in range(0, informants):
                self.informantIndices[j] = j
                links[memoryIndex][j] = j
                # print(links)
        elif (memoryIndex == 1):
            for j in range(4, swarm_size):
                self.informantIndices[col] = j
                links[memoryIndex][col] = j
                # print(str(memoryIndex) + " " + str(col) + " " + str(j))
                col = col + 1
        # print(links)

    # this method should be used by memory swarm
    def set_group_best(self, memoryIndex, swarm):
        best = 100
        best_index = -1
        # print("setting group best")
        # assume first is the best
        informantIndex = links[memoryIndex][0]
        informant_range = informantIndex + 4
        best = swarm[informantIndex].pbest
        for i in range(informantIndex+1, informant_range):
            i = informantIndex + 1
            if (swarm[i].pbest < best):
                best = swarm[i].pbest
                best_index = i
        self.group_best = best
        for j in range(0, dimensions):
            self.group_positions[j] = swarm[best_index].positions[j]

    # this method should be used by memory swarm
    def modify_group_best(self, memoryIndex, swarm):
        best_index = -1
        # print("modifying group best of memory swarm")
        # find the pbest among the explorers
        # if this is better than memory's group_best then update
        best = self.pbest
        informantIndex = links[memoryIndex][0]
        informant_range = informantIndex + 4
        for i in range(informantIndex+1, informant_range):
            i = informantIndex + 1
            if (swarm[i].pbest < best):
                best = swarm[i].pbest
                best_index = i
        if (best < self.group_best):
            self.group_best = best
            for j in range(0, dimensions):
                self.group_positions[j] = swarm[best_index].positions[j]

    def calculate_new_velocity(self, memoryIndex):
        for d in range(dimensions):
            part2 = random.uniform(0, cmax) * (self.best_positions[d] - self.positions[d])
            self.velocities[d] = c1 * self.velocities[d] + part2
            part3 = random.uniform(0, cmax) * (memory[memoryIndex].group_positions[d] - self.positions[d])
            self.velocities[d] = self.velocities[d] + part3

    def displayParticle(self):
        print("Particle details:")
        for i in range(0, self.dimensions):
            print("Positions:" + str(self.positions[i]) + " ")
            print("velocities: " + str(self.velocities[i]) + " ")
        print("fitness: " + str(self.fitness) + " pbest: " + str(self.pbest))
        # each particle's group best is not modified
        # + " group best : " + str(self.group_best)

    def move_memory_particle(self):
        if (self.group_best < self.pbest):
            for d in range(dimensions):
                part2 = random.uniform(0, cmax) * (self.best_positions[d] - self.positions[d])
                self.velocities[d] = c1 * self.velocities[d] + part2
                part3 = random.uniform(0, cmax) * (self.group_positions[d] - self.positions[d])
                self.velocities[d] = self.velocities[d] + part3
            for d in range(dimensions):
                if self.velocities[d] < min_boundary:
                    self.velocities[d] = min_boundary
                elif self.velocities[d] > max_boundary:
                    self.velocities[d] = max_boundary
            for d in range(dimensions):
                self.positions[d] += self.velocities[d]
        # print("memory moved")

# Initialize memory and explorer swarms
memory = [Particle(dimensions, min_boundary, max_boundary) for x in range(memory_size)]
swarm = [Particle(dimensions, min_boundary, max_boundary) for x in range(swarm_size)]

# calculate initial fitness and display particles
# the initial fitness is the best fitness initially
# the initial positions are the best positions initially
# set the group best of memory
for index, particle in enumerate(memory):
    particle.pbest = particle.calculateFitness()
    for d in range(particle.dimensions):
        particle.best_positions[d] = particle.positions[d]
    particle.set_informants(index)
    particle.set_group_best(index, swarm)
    # particle.displayParticle()

for e, particle in enumerate(swarm):
    particle.pbest = particle.calculateFitness()
    for d in range(particle.dimensions):
        particle.best_positions[d] = particle.positions[d]
    for index, mem in enumerate(memory):
        for j in range(0, informants):
            if (links[index][j] == e):
                # print("index " + str(index))
                particle.set_memories(index)
    # particle.displayParticle()

# repeatedly calculate fitness, compare memory's best
# update velocity and position till loop exits
for i in range(0, number_of_iterations):
    # modify the memory's group best based on the previous iteration
    for index, particle in enumerate(memory):
        particle.modify_group_best(index, swarm)
        # particle.displayParticle()
    # memory is used for only remembering the group best.
    for particle in swarm:
        # calculate new velocity
        # check whether the boundary is crossed
        # pass the index of memory swarm particle to access group best
        particle.calculate_new_velocity(particle.memoryIndices[0])
        for d in range(dimensions):
            if particle.velocities[d] < min_boundary:
                particle.velocities[d] = min_boundary
            elif particle.velocities[d] > max_boundary:
                particle.velocities[d] = max_boundary
        for d in range(dimensions):
            particle.positions[d] += particle.velocities[d]
            # check whether position is outside boundary
            if particle.positions[d] < min_boundary:
                particle.positions[d] = min_boundary
            elif particle.positions[d] > max_boundary:
                particle.positions[d] = max_boundary
        # compute fitness of the new positions
        particle.calculateFitness()
        # check and modify personal best
        if particle.fitness < particle.pbest:
            particle.pbest = particle.fitness
            particle.best_positions = list(particle.positions)
        # particle.displayParticle()
    memoryNum = 0
    for particle in memory:
        particle.move_memory_particle()
        particle.calculateFitness()
        if particle.fitness < particle.pbest:
            particle.pbest = particle.fitness
            particle.best_positions = list(particle.positions)
        memoryNum = memoryNum + 1
        # particle.displayParticle()
        print(" memory " + str(memoryNum) + " group best : " + str(particle.group_best))
# the results show that using informants and group best do not achieve
# the same convergence as using global best.
```

---

b.  Memory swarm particles move slower than the explorer swarm particle.  
    Introduce movement to the memory swarm particles.

**Answer Q1b:**

```python
# -*- coding: utf-8 -*-

"""
Modified to implement memory swarm movement (Question 1b).
Key changes:
1. Added inertia weight (w) and velocity updates for memory particles.
2. Boundary checks for memory particle positions/velocities.
"""

import random
import math

# Parameters (unchanged)
SWARM_SIZE_MAX = 40
MAX_DIMENSIONS = 10
swarm_size = 8
informants = 4
dimensions = 2
memory_size = 2
max_boundary = 2
min_boundary = -2
number_of_iterations = 10
c1 = 0.689343
cmax = 1.42694
w = 0.7  # Inertia weight for memory particles

class Particle:
    def __init__(self, dimensions, minValue, maxValue):
        # ... (existing code unchanged) ...
        pass  # Assuming other methods are unchanged

    def move_memory_particle(self):
        """
        Updated method for memory particle movement.
        Slower movement via reduced velocity scaling.
        """
        if self.group_best < self.pbest:
            for d in range(self.dimensions):
                # Cognitive component (pbest - current position)
                part_cognitive = random.uniform(0, cmax) * (self.best_positions[d] - self.positions[d])
                # Social component (group_best - current position)
                part_social = random.uniform(0, cmax) * (self.group_positions[d] - self.positions[d])
                # Update velocity with inertia (slower than explorers)
                self.velocities[d] = w * self.velocities[d] + part_cognitive + part_social
                # Enforce velocity boundaries
                self.velocities[d] = max(min(self.velocities[d], max_boundary), min_boundary)
            # Update position
            for d in range(self.dimensions):
                self.positions[d] += self.velocities[d] * 0.5  # Slower step size
                # Enforce position boundaries
                self.positions[d] = max(min(self.positions[d], max_boundary), min_boundary)
            # Recalculate fitness after movement
            self.calculateFitness()

# Initialize swarms (unchanged)
memory = [Particle(dimensions, min_boundary, max_boundary) for _ in range(memory_size)]
swarm = [Particle(dimensions, min_boundary, max_boundary) for _ in range(swarm_size)]

# Main loop (unchanged except for added memory movement)
for i in range(number_of_iterations):
    # ... (existing explorer swarm updates) ...
    # Memory swarm movement (Question 1b)
    for mem_particle in memory:
        mem_particle.move_memory_particle()
        if mem_particle.fitness < mem_particle.pbest:
            mem_particle.pbest = mem_particle.fitness
            mem_particle.best_positions = list(mem_particle.positions)
    # Print group bests for verification
    for idx, mem in enumerate(memory):
        print(f"Memory {idx} group best: {mem.group_best:.4f}")
```

---

**Q2)** What is the purpose of adaptive swarm? What are the parametric adaptation options?

**Answer Q2:**

```python
import numpy as np
import random

# Adaptive PSO for Alpine Function Optimization
def alpine_function(x):
    return sum(abs(x_i * np.sin(x_i) + 0.1 * x_i) for x_i in x)

# Adaptive PSO Parameters
SWARM_SIZE = 30
DIMENSIONS = 2
MAX_ITERATIONS = 100
MIN_BOUNDARY = -2
MAX_BOUNDARY = 2

class AdaptiveParticle:
    def __init__(self, dim):
        self.position = np.random.uniform(MIN_BOUNDARY, MAX_BOUNDARY, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.personal_learning_rate = 1.5  # c1 (adaptive)
        self.social_learning_rate = 1.5    # c2 (adaptive)

def run_adaptive_pso():
    # Initialize swarm
    swarm = [AdaptiveParticle(DIMENSIONS) for _ in range(SWARM_SIZE)]
    global_best_position = None
    global_best_fitness = float('inf')
    # Adaptive parameters
    initial_inertia = 0.9
    final_inertia = 0.4
    diversity_threshold = 0.1
    for iteration in range(MAX_ITERATIONS):
        # Calculate current inertia (linear decay)
        inertia = initial_inertia - (initial_inertia - final_inertia) * (iteration / MAX_ITERATIONS)
        # Calculate swarm diversity
        mean_position = np.mean([p.position for p in swarm], axis=0)
        diversity = np.mean([np.linalg.norm(p.position - mean_position) for p in swarm])
        # Adaptation rules
        for particle in swarm:
            if diversity < diversity_threshold:
                # Increase exploration
                particle.personal_learning_rate = min(2.0, particle.personal_learning_rate * 1.05)
                particle.social_learning_rate = max(0.5, particle.social_learning_rate * 0.95)
            else:
                # Increase exploitation
                particle.personal_learning_rate = max(0.5, particle.personal_learning_rate * 0.95)
                particle.social_learning_rate = min(2.0, particle.social_learning_rate * 1.05)
            # Update velocity and position
            r1, r2 = random.random(), random.random()
            cognitive = particle.personal_learning_rate * r1 * (particle.best_position - particle.position)
            social = particle.social_learning_rate * r2 * (global_best_position - particle.position) if global_best_position is not None else 0
            particle.velocity = inertia * particle.velocity + cognitive + social
            particle.position += particle.velocity
            # Apply bounds
            particle.position = np.clip(particle.position, MIN_BOUNDARY, MAX_BOUNDARY)
            # Evaluate fitness
            current_fitness = alpine_function(particle.position)
            # Update personal best
            if current_fitness < particle.best_fitness:
                particle.best_fitness = current_fitness
                particle.best_position = particle.position.copy()
            # Update global best
            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = particle.position.copy()
        print(f"Iter {iteration}: Fitness = {global_best_fitness:.4f}, Diversity = {diversity:.4f}")
    return global_best_position, global_best_fitness

# Run and display results
best_position, best_fitness = run_adaptive_pso()
print(f"
Optimized Solution: {best_position}")
print(f"Minimum Alpine Value: {best_fitness:.6f}")
```

---

**Q3)** Download the ANN source code files for the AND logic solution.  
Design a solution which incorporates PSO to optimize the weights of ANN.

**ANN Source Files;**

- **Model:**

```python
from collections import Counter
import numpy as np

# NumPy (Numerical Python) is an open source Python library.
# This contains multidimensional array and matrix data structures
# a wide variety of mathematical operations on arrays
# It provides ndarray, a homogeneous n-dimensional array object,
# with methods to efficiently operate on it.
class Neuron:
    def __init__(self, num_inputs, learning_rate):
        self.number_of_inputs = num_inputs
        self.weights = np.random.randn(1, 3)
        self.learning_rate = learning_rate
        self.bias = 1.0
        self.output = 0
        self.error = 0
        self.inputs = np.empty(num_inputs)  # empty array of len num_inputs

    def compute_output(x):
        if x <= 0:
            return 0
        else:
            return 1

    def __call__(self, input_data):
        data = np.concatenate((input_data, [self.bias]))
        result = self.weights @ data
        return Neuron.compute_output(result)

    def update_weight(self, target_result, in_data):
        ##check validity of types in the input data
        if type(in_data) != np.ndarray:
            in_data = np.array(in_data)
        calculated_result = self(in_data)  # invokes call
        error = target_result - calculated_result
        if error != 0:
            in_data = np.concatenate((in_data, [self.bias]))
            correction = error * in_data * self.learning_rate
            self.weights += correction

    def evaluate(self, data, labels):
        evaluation = Counter()
        for sample, label in zip(data, labels):
            result = self(sample)  # invokes call
            if result == label:
                evaluation["correct"] += 1
            else:
                evaluation["wrong"] += 1
        return evaluation

    def print_neuron(self):
        print(self.weights)
```

- **Data:**

```python
import numpy as np
from neuron_and_logic import Neuron
from collections import Counter

def labelled_samples(n):
    for _ in range(n):
        s = np.random.randint(0, 2, (2,))  # returns a tuple of 2 values in the range 0 and 1
        yield (s, 1) if s[0] == 1 and s[1] == 1 else (s, 0)

# num_inputs, learning_rate
p = Neuron(2, 0.2)  # weights are randomly calculated
for in_data, label in labelled_samples(30):
    p.update_weight(label, in_data)

test_data, test_labels = list(zip(*labelled_samples(30)))
evaluation = p.evaluate(test_data, test_labels)
print(evaluation)
# Counter({'correct': 30})
```

**Answer of Q3:**

- **neuron_pso.py (Modified Neuron Class)**

```python
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs + 1)  # +1 for bias
        self.fitness = float('inf')

    def compute_output(self, x):
        return 1 if x > 0 else 0  # Step activation

    def evaluate_fitness(self, data, labels):
        errors = 0
        for sample, label in zip(data, labels):
            # Append bias (1.0) to input
            input_with_bias = np.append(sample, 1.0)
            weighted_sum = np.dot(self.weights, input_with_bias)
            prediction = self.compute_output(weighted_sum)
            errors += (prediction != label)
        self.fitness = errors / len(data)  # Error rate (0-1)
        return self.fitness
```

- **pso_and_logic.py (PSO Training)**

```python
import numpy as np
from neuron_pso import Neuron
import random

# AND logic dataset
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

# PSO parameters
SWARM_SIZE = 20
MAX_ITERATIONS = 100
INERTIA = 0.7
COGNITIVE_WEIGHT = 1.5
SOCIAL_WEIGHT = 1.5

class Particle:
    def __init__(self, dim):
        self.position = np.random.rand(dim) * 2 - 1  # Random weights [-1, 1]
        self.velocity = np.random.rand(dim) * 0.1
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

def fitness_function(weights):
    neuron = Neuron(2)
    neuron.weights = weights
    return neuron.evaluate_fitness(data, labels)

# Initialize swarm
particles = [Particle(3) for _ in range(SWARM_SIZE)]  # 3 = 2 inputs + bias

global_best_position = None
global_best_fitness = float('inf')

# PSO main loop
for iteration in range(MAX_ITERATIONS):
    for particle in particles:
        # Evaluate current fitness
        current_fitness = fitness_function(particle.position)
        # Update personal best
        if current_fitness < particle.best_fitness:
            particle.best_fitness = current_fitness
            particle.best_position = particle.position.copy()
        # Update global best
        if current_fitness < global_best_fitness:
            global_best_fitness = current_fitness
            global_best_position = particle.position.copy()
    # Update velocities and positions
    for particle in particles:
        r1, r2 = random.random(), random.random()
        cognitive = COGNITIVE_WEIGHT * r1 * (particle.best_position - particle.position)
        social = SOCIAL_WEIGHT * r2 * (global_best_position - particle.position)
        particle.velocity = INERTIA * particle.velocity + cognitive + social
        particle.position += particle.velocity
    # Early stopping if perfect solution found
    if global_best_fitness == 0:
        break

# Results
print(f"Optimized Weights: {global_best_position}")
print(f"Classification Error: {global_best_fitness * 100:.2f}%")

# Test the trained ANN
neuron = Neuron(2)
neuron.weights = global_best_position
print("
AND Logic Test:")
for sample in data:
    prediction = neuron.compute_output(np.dot(neuron.weights, np.append(sample, 1.0)))
    print(f"Input: {sample} -> Output: {prediction}")
```