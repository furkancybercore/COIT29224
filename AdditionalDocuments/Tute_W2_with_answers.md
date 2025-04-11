# COIT29224 Evolutionary Computation Page | 1

## Tutorial Solution – Week 2

**Q1)** How do you define and measure the difficulty of an optimization problem?

**Answer:** The difficulty of an optimization problem in a given search space is the probability of not finding a solution by choosing a position at random according to a uniform distribution.

Difficulty can be expressed as 1 - failure probability.

**Q2)** How can you get the best results when a problem has more than one minimum such as local minimum and global minimum?

**Answer:**

Solutions around the global minimum can be obtained by reducing the tolerance level or the acceptable error. The other option is to reduce the search space and this requires some knowledge about the position of the solution.

**Q3)** Use the given source code file used in the lecture and understand the use of informants and group best in PSO.

a.  Try to vary the size of informants and observe the result.

b.  Compare the results with including a global best also.

**Answer (Code Analysis of Tute2Questn4.py):**

(a) The PSO implementation uses informants (local best within a subgroup) instead of global best.  
> Example: informants = 3 means each particle shares info with 3 neighbors.

(b) Result: Informants lead to slower convergence compared to global best (commented in the code), as local minima may trap particles.

**Q4)** Write a program to implement optimization of the Alpine function:

\[
f(x) = \sum_{d=1}^{D} x_d \sin(x_d) + 0.1\; x_d
\]

Search space is \([-2, 2]^{D}\). Optimize for \(D = 2\).

**Note:** Make necessary changes in the given source code for optimising \(f(x) = x^2 + y^2\). Make changes to the tolerance or the required precision. Observe the variations in results.

To create the \(random()\) values between -2 and +2 use the following relationship:

```
scaled value = min + (value * (max - min))
```

Where `min` and `max` are the minimum and maximum values of the desired range respectively, and `value` is the randomly generated floating point value in the range between 0 and 1.

**Answer Q4:**

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
x (t+1) = x(t) + v(t+1)
new velocity is calculated using the following relation
v(t+1) = wv(t) + c1*r1*(p(t)-x(t)) + C2*r2*(G - x(t))
p(t) is the particle's best and G is the global best
"""

import random
import math
import numpy as np

SWARM_SIZE_MAX = 40
MAX_DIMENSIONS = 10

swarm_size = 10  ## N
informants = 3   # K
dimensions = 2   # D

# LIENS list of informants
links = [[0 for v in range(SWARM_SIZE_MAX)] for v in range(SWARM_SIZE_MAX)]

TWO_PI = 6.283185307
E = 2.718281828

max_boundary = 2
min_boundary = -2

number_of_iterations = 100
c1 = 0.689343
cmax = 1.42694
desired_precision = 0.00001
fmin = 0  # fitness or objective to reach
w = 0.3   # inertia

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
        self.group_best = 100
        self.group_positions = [1.0 for d in range(dimensions)]

    def calculateFitness(self):
        f = 0
        for i in range(0, self.dimensions):
            f = f + self.positions[i] * math.sin(self.positions[i]) + 0.1 * self.positions[i]
        self.fitness = f
        return f

    def modifyPosition(self, new_positions):
        for i in range(0, self.dimensions):
            self.positions[i] = new_positions[i]

    def set_informants(self):
        for i in range(0, swarm_size):
            for j in range(0, informants):
                self.informantIndices[j] = random.randrange(0, swarm_size)

    def set_group_best(self, swarm):
        best = 100
        best_index = -1
        # print("setting group best")
        pbest = 100
        for i in range(0, informants):
            best = swarm[self.informantIndices[0]].pbest  # assume first is the best
            pbest = swarm[self.informantIndices[i]].pbest
            if (pbest < best):
                best = pbest
                best_index = i
        self.group_best = best
        for j in range(0, dimensions):
            self.group_positions[j] = swarm[best_index].positions[j]

    def modify_group_best(self, swarm):
        best = 100
        best_index = -1
        pbest = 100
        # print("modifying group best")
        for i in range(0, informants):
            best = swarm[self.informantIndices[0]].pbest
            pbest = swarm[self.informantIndices[i]].pbest
            if (pbest < best):
                best = pbest
        if (best < self.group_best):
            # print("best: " + str(best) + " group_best: " + str(self.group_best))
            self.group_best = best
            for j in range(0, dimensions):
                self.group_positions[j] = swarm[best_index].positions[j]

    def calculate_new_velocity(self):
        for d in range(dimensions):
            part2 = random.uniform(0, cmax) * (self.best_positions[d] - self.positions[d])
            self.velocities[d] = c1 * self.velocities[d] + part2
            part3 = random.uniform(0, cmax) * (self.group_positions[d] - self.positions[d])
            self.velocities[d] = self.velocities[d] + part3

    def displayParticle(self):
        print("Particle details:")
        for i in range(0, self.dimensions):
            print("Positions:" + str(self.positions[i]) + " ")
            print("velocities: " + str(self.velocities[i]) + " ")
        for d in range(0, informants):
            print("informants: " + str(self.informantIndices[d]))
        print("fitness: " + str(self.fitness) + " pbest: " + str(self.pbest) + " group best: " + str(self.group_best))

swarm = [Particle(dimensions, min_boundary, max_boundary) for x in range(swarm_size)]

# Calculate initial fitness and display particles
# The initial fitness is the best fitness initially
# The initial positions are the best positions initially
# Set informants and also set the group best
for particle in swarm:
    particle.pbest = particle.calculateFitness()
    for d in range(particle.dimensions):
        particle.best_positions[d] = particle.positions[d]
    particle.set_informants()
    particle.set_group_best(swarm)
    particle.displayParticle()

# Set the global best position and fitness
# Initially assume first is the best
# The global positions are not used to modify velocity.
# best_global_fitness = swarm[0].pbest
# for d in range(swarm[0].dimensions):
#     best_global_positions[d] = swarm[0].best_positions[d]
# for b in range(swarm_size):
#     if swarm[b].pbest < best_global_fitness:
#         best_global_fitness = swarm[b].pbest
#         for d in range(swarm[b].dimensions):
#             best_global_positions[d] = swarm[b].best_positions[d]

for i in range(0, number_of_iterations):
    for particle in swarm:
        # Calculate new velocity
        # Check whether the boundary is crossed
        # Modify the group best based on the previous iteration
        particle.modify_group_best(swarm)
        particle.calculate_new_velocity()
        for d in range(dimensions):
            if particle.velocities[d] < min_boundary:
                particle.velocities[d] = min_boundary
            elif particle.velocities[d] > max_boundary:
                particle.velocities[d] = max_boundary
        for d in range(dimensions):
            particle.positions[d] += particle.velocities[d]
            # Check whether position is outside boundary
            if particle.positions[d] < min_boundary:
                particle.positions[d] = min_boundary
            elif particle.positions[d] > max_boundary:
                particle.positions[d] = max_boundary
        # Compute fitness of the new positions
        particle.calculateFitness()
        # Check and modify personal best
        if particle.fitness < particle.pbest:
            particle.pbest = particle.fitness
            particle.best_positions = list(particle.positions)
        particle.displayParticle()
# The results show that using informants and group best do not achieve the same convergence as using global best.
```