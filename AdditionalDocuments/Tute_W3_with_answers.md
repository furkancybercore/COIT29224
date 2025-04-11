>Tutorial – Week 3

# 1.  What is meant by memory and explorer swarm?

## **Answer of Q1:**

  “
  Explorer the group of particles whose position and velocities are
  modified at each iteration, thus making them to move.

  The memory swarm is the group of particles who move only occasionally,
  towards the best position announced by the explorer swarm. This means
  that if the explorer swarm is not improving the performance, then
  there is no movement for the memory swarm.
  ”

# 2.  Explain the topologies available for informant structures.

## **Answer of Q2:**

  “
  There are two topologies for the informant structure. Fixed Topology:
  All the informants are defined once before the start of iteration and
  remains the same. In this all particles will have the same number of
  links. Random variable topology: The number of explorer particles
  connected to a memory swarm is decided at the beginning. With each
  iteration, each explorer establishes a link with a memory swarm.
  ”

# 3.  What are the proximity distributions used?

## **Answer of Q3:**

  “
  Velocity calculations are used for the movement of particles in a
  D-rectangle space using the equation:

  {𝑣𝑑 = 𝐶1 𝑣𝑑 + 𝑟𝑎𝑛𝑑(0,𝑐𝑚𝑎𝑥)(𝑃𝑑 − 𝑥𝑑) + 𝑟𝑎𝑛𝑑(0,𝑐𝑚𝑎𝑥)(𝑔𝑑 − 𝑣𝑑)
  
  𝑑 = 𝑥𝑑 + 𝑣𝑑

  The particles that have no influence are wrongly placed at the top of
  the D-rectangle.

  The proximity distributions include placing particles using other
  distributions such as sphere in a normal distribution, or Gaussian.
  ”

# 4.  You have implemented the program to optimize the Alpine function:

    f(x) = x sin(x) + 0.1x.

    a.  Introduce informants, if you haven’t done so.
    b.  Change the informants during each iteration?
    c.  Use memory swarm
    d.  How do we update the memory swarm’s movements?

## **Answer of Q4:**

```python
# -*- coding: utf-8 -*- 
""" 
Created on Friday Jul  10 16:55:02 2020 
@author: Mary Tom 
Reference: Particle Swarm Optimization - Clerk Chapter 6 
""" 
""" 
the cost function is the Alpine function 
Alpine 10D: ∑(𝑑=1^𝐷▒| X_𝑑 sin( 𝑥_𝑑)+0.1 X_d | 
f(x) = for d=1 to D sum abs((x sin(x)+ 0.1x)) where d is the dimension. 
for a two dimension, it is x1 sin(x1)+0.1x1+ x2 sin(x2)+0.1x2 
Minimize f(x) where the boundaries are -2 to 2 for x. 
the lower and upper boundary of x  dimension is -2 and 2 
new position x is calculated from the previous and new velocity 
x (t+1 )   = x(t) +v(t+1) 
new velocity is calculated using the following relation 
v(t+1)  = wv (t)+ c1r1(p(t)-x(t)) + C2r2 (G- x(t)) 
p(t) is the particle's best and G is the global best 
This program uses one memory particle following the star topology.  
This is found to be the most effective for Alpine function. 
""" 
import random 
import math 
#import numpy as np 
SWARM_SIZE_MAX = 40    
MAX_DIMENSIONS = 10 
swarm_size = 10             
informants = 3 #  K 
dimensions = 2   # D 
##  N 
#LINKS list of informants 
links = [[0 for v in range(SWARM_SIZE_MAX)] for v in range(SWARM_SIZE_MAX)] 
TWO_PI = 6.283185307 
E  = 2.718281828 
max_boundary = 2 
min_boundary = -2 
number_of_iterations= 10 
c1 = 0.689343 
cmax = 1.42694 
desired_precision = 0.00001 
fmin =0  #fitness or objective to reach 
w = 0.7 # inertia 
best_global_positions =[1.8 for v in range(dimensions)] 
best_global_fitness = 800 
############################################################ 
class Particle: 
def __init__(self, dimensions, minValue, maxValue): 
self.dimensions = dimensions 
self.min_boundary = minValue 
self.max_boundary = maxValue 
self.positions = [1 for d in range (dimensions)] 
self.velocities = [1 for d in range(dimensions)] 
#init x and y values 
#self.informantIndices =[0 for i in range(informants)] 
# use one memory-swarm to inform following star topology 
for j in range(0,dimensions): 
value = random.random() 
scaled_value = min_boundary + (value*(max_boundary - min_boundary)) 
self.positions[j] = scaled_value 
#init velocities of x and y 
value = random.random() 
scaled_value = min_boundary/2 + (value*((max_boundary - min_boundary)/2  - (min_boundary-max_boundary)/2 ) ) 
self.velocities[j] = scaled_value 
# maximum fitness of self 
self.fitness = 100 
self.pbest = 100   
self.best_positions = [1.0 for v in range(dimensions)] 
self.group_best = 100 # this value is only used for the memory particle 
self.group_positions = [1.0 for d in range(dimensions)] 
def calculateFitness(self): 
f = 0; 
for i in range (0, self.dimensions): 
f = f + abs (self.positions[i]*math.sin(self.positions[i])+0.1*self.positions[i]) 
self.fitness = f 
return f 
def modifyPosition(self, new_positions): 
for i in range(0, self.dimensions): 
self.positions[i] = new_positions[i] 
# this is method for setting informants from the same swarm 
def set_informants(self): 
for i in range(0, swarm_size): 
for j in range(0, informants): 
self.informantIndices[j] = random.randrange(0, swarm_size) 
#this method should be used by memeory swarm 
def set_group_best(self,swarm): 
best = 100 
best_index = -1 
#     print("setting group best") 
self.pbest = 100 
#assume first is the best 
best = swarm[0].pbest 
for i in range(1, swarm_size): 
if ( swarm[i].pbest < best): 
best = swarm[i].pbest 
best_index = i 
self.group_best = best 
for j in range(0, dimensions): 
self.group_positions[j] = swarm[best_index].positions[j] 
#this method should be used by memeory swarm 
def modify_group_best (self,swarm): 
#best = 100 
best_index = -1 
#      print("modifying group best of memeory swarm") 
# find the pbest among the explorers 
# if this is better than memeory's group_best then update 
best = swarm[0].pbest 
for i in range(1, swarm_size): 
if (swarm[i].pbest < best): 
best = swarm[i].pbest 
best_index = i 
#   best = pbest 
if (best < self.group_best): 
self.group_best = best 
#            print("best: " +str(best) + " group_best: " + str(self.group_best)) 
for j in range(0, dimensions): 
self.group_positions[j] = swarm[best_index].positions[j] 
def calculate_new_velocity(self): 
for d in range(dimensions): 
part2 = random.uniform(0, cmax)*(self.best_positions[d] - self.positions[d]) 
self.velocities[d] = c1* self.velocities[d] +part2  
part3 = random.uniform(0, cmax)*(memory.group_positions[d] - self.positions[d]) 
self.velocities[d] = self.velocities[d] + part3  
def displayParticle(self): 
print("Particle details:") 
for i in range (0, self.dimensions): 
print ("Positions:"+str(self.positions[i])+"    ") 
print ("velocities: " +str(self.velocities[i])+ "  ") 
print ("fitness: "+ str(self.fitness) + "pbest: "+str(self.pbest)  
+" group best : "+str(self.group_best))  
memory = Particle (dimensions, min_boundary, max_boundary) 
swarm = [Particle(dimensions, min_boundary, max_boundary)  
for x in range (swarm_size)] 
#calculate initial fitness and display particles 
#the initial fitness is the best fitness initially 
#the initial positions are the best positions initially 
#set the group best of memory 
memory.pbest = memory.calculateFitness() 
for d in range(memory.dimensions): 
memory.best_positions[d] = memory.positions[d] 
memory.set_group_best(swarm) 
memory.displayParticle() 
for particle in swarm: 
particle.pbest = particle.calculateFitness() 
for d in range(particle.dimensions): 
particle.best_positions[d] = particle.positions[d] 
particle.displayParticle() 
# rpeatedly calculate fitness, compaore memeory's best 
# update velocity and position till loop exits  
for i in range(0, number_of_iterations): 
#modify the memory's group best based on the previous iteration 
memory.modify_group_best(swarm) 
memory.displayParticle() 
# memory is used for only remembering the group best. 
for particle in swarm: 
#calculate new velocity 
#check whether the boundary is crossed 
particle.calculate_new_velocity() 
for d in range(dimensions): 
if  particle.velocities[d] < min_boundary: 
particle.velocities[d] = min_boundary 
elif particle.velocities[d] > max_boundary: 
particle.velocities[d] = max_boundary 
for d in range(dimensions): 
particle.positions[d] += particle.velocities[d] 
#check whether position is outside boundary 
if particle.positions[d] <  min_boundary: 
particle.positions[d] = min_boundary 
elif particle.positions[d] > max_boundary: 
particle.positions[d] = max_boundary 
#compute fitness of the new positions 
particle.calculateFitness() 
#check and modify personal best  
if particle.fitness < particle.pbest: 
particle.pbest = particle.fitness 
particle.best_positions = list(particle.positions) 
particle.displayParticle() 
memory.displayParticle() 
#the results show that using informants and group best do not achieve the same convergence 
as using global best. 
```

# PSO Demo;

```python
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 18 13:23:22 2025
@author: umair
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# PSO parameters
SWARM_SIZE = 20  # Number of particles
DIMENSIONS = 2   # Number of dimensions
INFORMANTS = 3   # Number of informants for group best
NUM_GENERATIONS = 50  # Number of iterations
W = 0.729  # Inertia weight
C1 = 1.49  # Cognitive coefficient
C2 = 1.49  # Social coefficient
MIN_BOUNDARY = -100
MAX_BOUNDARY = 100

# Function to minimize
def fitness_function(position):
    x, y = position
    return x - y + 7  # Modify as needed

class Particle:
    def __init__(self):
        self.position = [random.uniform(MIN_BOUNDARY, MAX_BOUNDARY) for _ in range(DIMENSIONS)]
        self.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
        self.fitness = fitness_function(self.position)
        self.best_position = list(self.position)
        self.best_fitness = self.fitness
        self.informants = random.sample(range(SWARM_SIZE), INFORMANTS)  # Select informants randomly
        self.group_best_position = list(self.position)
        self.group_best_fitness = self.fitness

    def update_velocity(self, global_best_position):
        for d in range(DIMENSIONS):
            r1, r2 = random.random(), random.random()
            cognitive_component = C1 * r1 * (self.best_position[d] - self.position[d])
            social_component = C2 * r2 * (self.group_best_position[d] - self.position[d])
            self.velocity[d] = W * self.velocity[d] + cognitive_component + social_component

    def update_position(self):
        for d in range(DIMENSIONS):
            self.position[d] += self.velocity[d]
            # Boundary constraints
            self.position[d] = max(min(self.position[d], MAX_BOUNDARY), MIN_BOUNDARY)
        self.fitness = fitness_function(self.position)

    def update_group_best(self, swarm):
        """ Updates the group best position based on informants' best fitness. """
        best_informant = min(self.informants, key=lambda i: swarm[i].best_fitness)
        if swarm[best_informant].best_fitness < self.group_best_fitness:
            self.group_best_fitness = swarm[best_informant].best_fitness
            self.group_best_position = list(swarm[best_informant].best_position)

# Initialize swarm
swarm = [Particle() for _ in range(SWARM_SIZE)]

# Initialize global best
global_best_particle = min(swarm, key=lambda p: p.best_fitness)
global_best_position = list(global_best_particle.best_position)
global_best_fitness = global_best_particle.best_fitness

# Visualization setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(MIN_BOUNDARY, MAX_BOUNDARY)
ax.set_ylim(MIN_BOUNDARY, MAX_BOUNDARY)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("PSO Optimization with Group Bests")

# Generate contour plot for visualization
X, Y = np.meshgrid(np.linspace(MIN_BOUNDARY, MAX_BOUNDARY, 100), np.linspace(MIN_BOUNDARY, MAX_BOUNDARY, 100))
Z = X - Y + 7  # Modify if using a different fitness function
ax.contourf(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.5)

# Plot particles
positions = np.array([p.position for p in swarm])
scat = ax.scatter(positions[:, 0], positions[:, 1], c='red', label="Particles", edgecolors="black")
global_best_marker, = ax.plot([], [], 'bo', markersize=10, label="Global Best")
plt.legend()
plt.ion()  # Enable interactive mode

# Main PSO loop
for generation in range(NUM_GENERATIONS):
    for particle in swarm:
        particle.update_group_best(swarm)  # Update group best based on informants
        particle.update_velocity(global_best_position)
        particle.update_position()
        # Update personal best
        if particle.fitness < particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_position = list(particle.position)
    # Update global best
    best_particle = min(swarm, key=lambda p: p.best_fitness)
    if best_particle.best_fitness < global_best_fitness:
        global_best_fitness = best_particle.best_fitness
        global_best_position = list(best_particle.best_position)
    # Update visualization
    positions = np.array([p.position for p in swarm])
    scat.set_offsets(positions)
    if len(global_best_position) == DIMENSIONS:
        global_best_marker.set_data([global_best_position[0]], [global_best_position[1]])
    ax.set_title(f"Iteration {generation + 1}")
    plt.draw()
    plt.pause(0.5)
plt.ioff()  # Turn off interactive mode
plt.show()

# Final results
print("
Optimization Complete!")
print(f"Best Position: {global_best_position}")
print(f"Best Fitness Value: {global_best_fitness}")
```

# PSO Notes

## ✅ General Structure (Strategically Same)

Most bio-inspired metaheuristics (Algorithms designed to find good enough solutions for complex optimization problems—especially when traditional methods (like brute force or exact algorithms) are too slow or infeasible due to problem size or complexity. e.g., Genetic Algorithms, Particle Swarm Optimization, Ant Colony Optimization, etc.) do indeed share a common high-level structure, which typically includes:

1. Initialization: Generate an initial population (or solution set).

2. Evaluation: Assess the fitness or quality of each solution.

3. Optimization Loop:
    - Selection or influence-based operations (e.g., selecting parents in GA, social influence in PSO).
    - Variation/Update (e.g., crossover/mutation in GA, velocity updates in PSO).
    - Replacement or update of the population.
    - Evaluation again.

4. Termination: Stop after a certain number of iterations or when convergence is achieved.

This abstract loop applies to nearly all population-based metaheuristics.

Hence, strategically, they are the same:
“Start with a bunch of guesses → improve guesses iteratively → pick the best.”

## ⚙️ Tactical Differences

Where they differ is in their inspiration, operators, mechanisms, and mathematical modeling — i.e., the tactics. For example:
- GA: Uses crossover/mutation based on natural selection.
- PSO: Updates particles using cognitive and social terms.
- ACO: Simulates pheromone-based path finding.
- DE (Differential Evolution): Perturbs solutions using scaled differences.

So, tactically, they behave differently — in terms of exploration/exploitation balance, convergence speed, and problem suitability.

## ✅ Summary

Yes, you can say:
"All bio-inspired metaheuristics are strategically the same but tactically different."
This is actually a very useful way to think about them when teaching, explaining, or even designing new algorithms (or hybrids).

## ✅ Strategic View of PSO (Particle Swarm Optimization)

| Strategic Step | PSO Implementation |
|----------------|--------------------|
| **1. Initialization** | Initialize a swarm of particles with random positions and velocities in the solution space. |
| **2. Evaluation** | Evaluate the fitness (objective function value) of each particle. |
| **3. Optimization Loop** | For each iteration: |
|  | - Update velocities based on: |
|  |   - Particle’s own best-known position (\( pBest \)) |
|  |   - Swarm’s globally best-known position (\( gBest \)) |
|  | - Update positions using the new velocities |
|  | - Evaluate fitness of new positions |
|  | - Update \( pBest \) and \( gBest \) if improvements are found |
| **4. Termination** | Stop after a set number of iterations or if the solution converges. |
| **5. Output** | Return the globally best solution found by the swarm. |

## 🧩 Summary: PSO is Strategically Common, Tactically Unique

- Strategically same as other metaheuristics: it follows the population-based search model.
- Tactically different through its use of velocities, personal and social memory, and influence dynamics rather than crossover/mutation or pheromones.

>graph TD
>    A["Initialization of PSO parameters, random initialization of particle position and velocity"] --> B["Evaluate >the Fitness Function for each particle for local and global best solution"]
>    B --> C["Update the velocity of each particle"]
>    C --> D["Update the position of each particle"]
>    D --> E{"Is the stopping criteria satisfied?"}
>    E -- Yes --> F["Stop"]
>    E -- No --> G["Time iteration t = t + 1"]
>    G --> B


>graph TD
>    A["Initialize Input Data"] --> B["Calculate the objective Function"]
>    B --> C["Determine Pbest"]
>    C --> D["Determine Gbest"]
>    D --> E["Update Velocity and position"]
>    E --> F{"Check Stop Condition"}
>    F -- Yes --> G["Output = Gbest"]
>    F -- No --> H["Increase Iteration number"]
>    H --> B

## City Visit Problem Example

You have one day in a city and 5 cool places to visit:

1. Museum
2. Art Gallery
3. City Park
4. Science Center
5. Local Market

You can choose how much time to spend at each place.
Spending more time at a place means you enjoy it more, but it also costs more.
Your goal is to visit as many places as possible while not spending too much money.

### ✅ Decision Variables

Let:
- a1: fraction of time at Museum
- a2: fraction of time at Art Gallery
- a3: fraction of time at City Park
- a4: fraction of time at Science Center
- a5: fraction of time at Local Market

Each \( a_i \in [0, 1] \)

### 💸 Cost Info

| Place             | Time (min) | Cost Rate (¢/min) | Full Visit Cost (¢) |
|-------------------|------------|--------------------|--------------------|
| Museum            | 80         | 18.75              | 1500               |
| Art Gallery       | 60         | 16.67              | 1000               |
| City Park         | 40         | 12.5               | 500                |
| Science Center    | 70         | 17.14              | 1200               |
| Local Market      | 50         | 16.00              | 800                |


### 🧮 Objective Function

You want to maximize enjoyment per cost.
So we reward you for spending time at each place, but penalize expensive ones more.

Final Objective to Maximize:
Z = 0.7a_1 + 0.8a_2 + 0.9a_3 + 0.76a_4 + 0.82a_5

These weights are based on how cost-effective each attraction is:
- Cheaper = higher reward
- Expensive = lower reward

## Use PSO to solve this:

### Strategic Step 1: Initialization

- Inertia weight w = 0.5
- Cognitive and Social parameters c1 = c2 = 1.0
- Random numbers r1 = r2 = 0.5 (fixed for simplicity)
- Initial velocity: 0 for all dimensions
- Number of iterations: 2
- Swarm size: 3


Initial Positions and Velocities

| Particle | Initial Position \([a_1, a_2, a_3, a_4, a_5]\) | Velocity (All = 0)    |
|----------|-------------------------------------------------|-----------------------|
| P1       | [0.6, 0.3, 0.9, 0.5, 0.4]                       | [0, 0, 0, 0, 0]      |
| P2       | [0.2, 0.6, 0.5, 0.4, 0.7]                       | [0, 0, 0, 0, 0]      |
| P3       | [0.7, 0.4, 0.3, 0.6, 0.5]                       | [0, 0, 0, 0, 0]      |


### Strategic Step 2: Evaluation

#### Objective Calculation:
\[ 
Z = 0.7a_1 + 0.8a_2 + 0.9a_3 + 0.76a_4 + 0.82a_5 
\]

- **P1:**
  \[ 
  Z_1 = 0.42 + 0.24 + 0.81 + 0.38 + 0.328 = 2.178 
  \]

- **P2:**
  \[ 
  Z_2 = 0.14 + 0.48 + 0.45 + 0.304 + 0.574 = 1.948 
  \]

- **P3:**
  \[ 
  Z_3 = 0.49 + 0.32 + 0.27 + 0.456 + 0.41 = 1.946 
  \]

**pBest (personal best):** initial positions  
**gBest (global best):** P1, score = 2.178


### Strategic Step 3: Optimization Loop


#### Iteration 1

##### Update Velocities and Positions

Using:
\[ 
v_i = w \cdot v_i + c_1 \cdot r_1(pBest_i - x_i) + c_2 \cdot r_2(gBest_i - x_i) 
\]

Let’s perform all calculations for P2 and P3 (P1 won't move in this iteration because \( pBest = gBest = \text{position} \)).


##### Particle 1 (P1)

- **Initial Position:** [0.6, 0.3, 0.9, 0.5, 0.4]  
- **Velocity:** [0, 0, 0, 0, 0]  
- **pBest:** current  
- **gBest:** current  

###### **Step 1:** 

Velocity Update (since \( pBest = x \) and \( gBest = x \), all updates = 0). Position remains: [0.6, 0.3, 0.9, 0.5, 0.4]  

###### **Step 2:** Fitness Evaluation 
 
\[ 
Z = 0.7(0.6) + 0.8(0.3) + 0.9(0.9) + 0.76(0.5) + 0.82(0.4) 
\]  
\[ 
= 0.42 + 0.24 + 0.81 + 0.38 + 0.328 = 2.178 
\]  

pBest remains the same.


##### Particle 2 (P2)

- **Initial Position:** [0.2, 0.6, 0.5, 0.4, 0.7]  
- **Velocity:** [0, 0, 0, 0, 0]  
- **pBest:** current  
- **gBest:** [0.6, 0.3, 0.9, 0.5, 0.4]  

###### **Step 1:** Velocity Update (dimension-wise)

| i | \( x_i \) | \( gBest_i \) | Velocity Update                       | New \( x_i \) |
|---|-----------|----------------|--------------------------------------|---------------|
| 1 | 0.2       | 0.6            | \( 0 + 0 + 0.5(0.6 - 0.2) = 0.2 \)  | 0.4           |
| 2 | 0.6       | 0.3            | \( 0 + 0 + 0.5(0.3 - 0.6) = -0.15 \)| 0.45          |
| 3 | 0.5       | 0.9            | \( 0 + 0 + 0.5(0.9 - 0.5) = 0.2 \)  | 0.7           |
| 4 | 0.4       | 0.5            | \( 0 + 0 + 0.5(0.5 - 0.4) = 0.05 \) | 0.45          |
| 5 | 0.7       | 0.4            | \( 0 + 0 + 0.5(0.4 - 0.7) = -0.15 \)| 0.55          |

**Updated Position:** [0.4, 0.45, 0.7, 0.45, 0.55]  

###### **Step 2:** Fitness Evaluation  

\[ 
Z = 0.7(0.4) + 0.8(0.45) + 0.9(0.7) + 0.76(0.45) + 0.82(0.55) 
\]  
\[ 
= 0.28 + 0.36 + 0.63 + 0.342 + 0.451 = 2.063 
\]  

- \( 2.063 > 1.948 \) → Update \( pBest \)  
- New \( pBest = [0.4, 0.45, 0.7, 0.45, 0.55] \)  
- \( pBest_Z = 2.063 \)


##### Particle 3 (P3)

- **Initial Position:** [0.7, 0.4, 0.3, 0.6, 0.5]  
- **Velocity:** [0, 0, 0, 0, 0]  
- **pBest:** current  
- **gBest:** [0.6, 0.3, 0.9, 0.5, 0.4]  

###### **Step 1:** Velocity Update

| i | \( x_i \) | \( gBest_i \) | Velocity Update                       | New \( x_i \) |
|---|-----------|----------------|--------------------------------------|---------------|
| 1 | 0.7       | 0.6            | \( 0 + 0 + 0.5(0.6 - 0.7) = -0.05 \)| 0.65          |
| 2 | 0.4       | 0.3            | \( 0 + 0 + 0.5(0.3 - 0.4) = -0.05 \)| 0.35          |
| 3 | 0.3       | 0.9            | \( 0 + 0 + 0.5(0.9 - 0.3) = 0.3 \)  | 0.6           |
| 4 | 0.6       | 0.5            | \( 0 + 0 + 0.5(0.5 - 0.6) = -0.05 \)| 0.55          |
| 5 | 0.5       | 0.4            | \( 0 + 0 + 0.5(0.4 - 0.5) = -0.05 \)| 0.45          |

**Updated Position:** [0.65, 0.35, 0.6, 0.55, 0.45]  

###### **Step 2:** Fitness Evaluation  

\[ 
Z = 0.7(0.65) + 0.8(0.35) + 0.9(0.6) + 0.76(0.55) + 0.82(0.45) 
\]  
\[ 
= 0.455 + 0.28 + 0.54 + 0.369 = 2.062 
\]  

- \( 2.062 > 1.946 \) → Update \( pBest \)  
- New \( pBest = [0.65, 0.35, 0.6, 0.55, 0.45] \)  
- \( pBest_Z = 2.062 \)



##### End of Iteration 1 Summary

| Particle | Position                | Z     | pBest Updated? |
|----------|-------------------------|-------|-----------------|
| P1       | [0.6, 0.3, 0.9, 0.5, 0.4] | 2.178 | No              |
| P2       | [0.4, 0.45, 0.7, 0.45, 0.55] | 2.063 | Yes             |
| P3       | [0.65, 0.35, 0.6, 0.55, 0.45] | 2.062 | Yes             |

- **gBest still = P1**




#### Iteration 2

##### Particle 1 (P1)

- **Position:** [0.6, 0.3, 0.9, 0.5, 0.4]  
- **Velocity:** [0, 0, 0, 0, 0]  
- **No change since \( pBest = gBest = \text{current} \)**  

**Z:** 2.178 (unchanged)


##### Particle 2 (P2)

- **Previous Position:** [0.4, 0.45, 0.7, 0.45, 0.55]  
- **Velocity:** [0.2, -0.15, 0.2, 0.05, -0.15]  
- **gBest:** [0.6, 0.3, 0.9, 0.5, 0.4]  

**Velocity Update**

| i | \( v_i \) | \( x_i \) | \( gBest_i \) | \( v_{new} \)                     | \( x_{new} \)                   |
|---|-----------|-----------|----------------|-----------------------------------|----------------------------------|
| 1 | 0.2       | 0.4       | 0.6            | \( 0.1 + 0.1 = 0.2 \)             | \( 0.4 + 0.2 = 0.6 \)           |
| 2 | -0.15     | 0.45      | 0.3            | \( -0.075 - 0.075 = -0.15 \)      | \( 0.45 - 0.15 = 0.3 \)         |
| 3 | 0.2       | 0.7       | 0.9            | \( 0.1 + 0.1 = 0.2 \)             | \( 0.7 + 0.2 = 0.9 \)           |
| 4 | 0.05      | 0.45      | 0.5            | \( 0.025 + 0.025 = 0.05 \)        | \( 0.45 + 0.05 = 0.5 \)         |
| 5 | -0.15     | 0.55      | 0.4            | \( -0.075 - 0.075 = -0.15 \)      | \( 0.55 - 0.15 = 0.4 \)         |

**New Position:** [0.6, 0.3, 0.9, 0.5, 0.4]  

**Z:** 2.178 → \( pBest \) updated


##### Particle 3 (P3)

- **Previous Position:** [0.65, 0.35, 0.6, 0.55, 0.45]  
- **Velocity:** [-0.05, -0.05, 0.3, -0.05, -0.05]  
- **gBest:** [0.6, 0.3, 0.9, 0.5, 0.4]  

**Velocity Update**

| i | \( v_i \) | \( x_i \) | \( gBest_i \) | \( v_{new} \)                     | \( x_{new} \)                   |
|---|-----------|-----------|----------------|-----------------------------------|----------------------------------|
| 1 | -0.05     | 0.65      | 0.6            | \( -0.025 - 0.025 = -0.05 \)      | \( 0.65 - 0.05 = 0.6 \)         |
| 2 | -0.05     | 0.35      | 0.3            | \( -0.025 - 0.025 = -0.05 \)      | \( 0.35 - 0.05 = 0.3 \)         |
| 3 | 0.3       | 0.6       | 0.9            | \( 0.15 + 0.15 = 0.3 \)           | \( 0.6 + 0.3 = 0.9 \)           |
| 4 | -0.05     | 0.55      | 0.5            | \( -0.025 - 0.025 = -0.05 \)      | \( 0.55 - 0.05 = 0.5 \)         |
| 5 | -0.05     | 0.45      | 0.4            | \( -0.025 - 0.025 = -0.05 \)      | \( 0.45 - 0.05 = 0.4 \)         |

**New Position:** [0.6, 0.3, 0.9, 0.5, 0.4]  

**Z:** 2.178 → \( pBest \) updated


#### Final Result After Iteration 3 (No Calculations)

All particles have converged to:

\[
[a_1, a_2, a_3, a_4, a_5] = [0.6, 0.3, 0.9, 0.5, 0.4]
\]

\[
Z = 0.7 \times 0.6 + 0.8 \times 0.3 + 0.9 \times 0.9 + 0.76 \times 0.5 + 0.82 \times 0.4 = 2.178
\]
