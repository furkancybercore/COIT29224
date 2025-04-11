# Tutorial – Week 1 Python Programming

1.  Write a program to calculate the tax payable based on the income.

    Write a method named `calculateTax()` which takes the user name and
    income entered by the user and calculates and displays the tax till
    the user enters -1 to quit. The income tax for income up to $17,000 is
    zero or nil; $17,000 – 35,000 is 12% of the amount over $17,000 and
    for income $35,000 – 85,000 is 30% of the amount over $35,000.

    **Answer code:**

    ```python
    # -*- coding: utf-8 -*-

    """
    Created on Sun Jul 5 17:54:54 2020
    @author: Mary Tom
    This is a program to calculate tax payable
    Solution to Tutorial Week One Question 1
    This demonstrates the use of input to display prompt and read user
    entry from keyboard.
    This program also demosntrates use of various control constructs.
    """

    def calculateTax():

        name = input("Enter your name: ")

        tax = 0

        income = int(input("enter your income: to finish input enter -1"))

        while income != -1:

            if income > 17000 and income <= 35000:
                tax = 0.12 *(income - 17000)
            elif income > 35000 and income < 85000:
                tax = 0.30 * (income - 35000)

            print ("Hello " + name + " your tax payable is: " + str(tax))
            income = int(input("enter your income: to finish input enter -1"))

    calculateTax()
    ```

2.  Write a program to calculate and display values in the range 2 to 24 in steps of 3.

    **Answer code:**

    ```python
    # -*- coding: utf-8 -*-

    """
    Created on Sun Jul 5 20:23:09 2020
    @author: Mary Tom
    Solution to Tutorial Week One Question 2
    demonstration of for
    """

    def display_Values_InRange():
        for n in range(2, 24,3):
            print("the value is: " + str(n))

    display_Values_InRange()
    ```

3.  Write a program that stores the names “Ron”, “Joy”, “Raj”, “Pat”, “Jay” in a list and displays the values and index.

    **Answer code:**

    ```python
    # -*- coding: utf-8 -*-

    """
    Created on Sun Jul 5 20:34:15 2020
    @author: Mary Tom
    Solution Week Tutorial Question 3
    """

    def display_list():
        names = ["Ron", "Joy", "Raj", "Pat", "Jay"]
        for i in range(len(names)):
            print(i, names[i])

    display_list()
    ```

4.  Create a class called Invoice that has the attributes part number, part description, quantity of items purchased, and price per item. The Invoice class have an init method, a function to display the invoice, and a function to calculate and return the invoice amount which is quantity multiplied by the price per item.

    **Answer code:**

    ```python
    """
    Created on Sun Jul 5 20:45:11 2020
    @author: MaryTom
    Solution to Tutorial Question 4 Invoice class
    file name: Q4invliceClass.py
    """

    class Invoice:
        def __init__(self, part_number, description, quantity, price ):
            self.part_number = part_number
            self.part_description = description
            self.quantity = quantity
            self.price = price

        def calculate_invoice_amount(self):
            amount = float(self.quantity * self.price)
            return amount

        def display_invoice(self):
            print("Invoice details:")
            print("part_number:" + self.part_number + " " + "description:" + self.part_description)
            print("quantity: " + str(self.quantity))
            print("amount: " + str(self.calculate_invoice_amount()))

    inv = Invoice("10", "hard disk", 2, 250)
    inv.display_invoice()
    ```

## Particle Swarm Optimization

5.  What are the constructs used to model a PSO solution? Define each of the elements involved such as swarm size, pbest, gbest, global best, velocity, and confidence levels.

    **Answer:**

    PSO is based on bird flocking in a two-dimensional space. Modeling a solution uses swarm as the flock of birds and particle as each agent. The particle’s xy position and velocity in the direction of xy axes are kept. The particle is moved by changing the velocity and thereby the position. The dimension of the search space is the number of variables in the target function to be optimized.

    Particle improves its movement towards the target by comparing to its own best, and the swarm best, or the group best.

    - **Swarm size** is the number of particles used to represent the flock of birds.
    - **pbest** is the best position so far taken by a particle during its movement.
    - **gbest** is the local best or the best within a small group of particles.
    - **Global best** is the best out of the whole swarm.
    - **Velocity** is represented by two values of x and y indicating the velocity in x direction and that in y direction.
    - **C1:** confidence in the particle’s own movement.
    - **C2:** confidence in its best performance.
    - **C3:** confidence in its informant’s best performance. Informants are the randomly selected neighbours.

6.  What is meant by the term explosion, and how this can be avoided?

    **Answer:**

    If the velocity is kept increasing, there is a potential for the particle to leave the search space, which is known as explosion. The confidence level c1 is kept below 1 to avoid this. Otherwise, a value for Vmax can be set and then particles can be tested for velocity not greater than Vmax.

7.  What are the parameters to be set and the values that can be used?

    **Answer:**

    - **N:** Swarm size: 20 – 40
    - **K:** Group size of informant: 3-5
    - **C1:** Self-confidence: 0-1, can be 0.7
    - **C2:** Confidence in others: about 1.5 (can be 1.43)
    - **Vmax:** Maximum velocity 0.5*(Xmax - Xmin)

8.  Run the program using the source code given for the example used in the Lecture slides. Observe the results.

    a.  What are the parameters used to change velocity?

    **Answer:**

    The velocity in PSO is updated using:
    - Inertia weight (`w`) – Controls momentum (e.g., `w = 0.729`).
    - Cognitive coefficient (`c1`) – Confidence in the particle’s best position (e.g., `c1 = 1.49`).
    - Social coefficient (`c2`) – Confidence in the swarm’s best position (e.g., `c2 = 1.49`).
    - Random factors (`r1`, `r2`) – Stochastic components.

    **Source:**
    - Defined in `pso.py` and `pso_animation.py` (lines: `w = 0.729`, `c1 = 1.49`, `c2 = 1.49`).
    - Velocity update formula:

    ```python
    self.velocity[d] = w * self.velocity[d] + c1*r1*(pbest - position) + c2*r2*(gbest - position)
    ```

    a.  How can this be changed to have local best to modify velocity?

    **Answer:**

    Replace `gbest` (global best) with `lbest` (local best, i.e., best in a neighborhood).

    - **Implementation:**
      1. Define a neighborhood (e.g., ring topology).
      2. Track `lbest` for each particle (best in its neighborhood).
      3. Modify the velocity update to use `lbest` instead of `gbest`.

    **Example Code Change (in `pso.py`):**

    ```python
    # Replace:
    social_component = c2 * r2 * (global_best_position[d] - self.position[d])
    # With:
    social_component = c2 * r2 * (local_best_position[d] - self.position[d])
    ```

    a.  Change the swarm size, and the confidence levels, each change being made one at a time.

    **Answer:**

    - **Swarm size:** Modify `swarm_size` (e.g., `swarm_size = 20`).
    - **Confidence levels:** Adjust `c1` and `c2` (e.g., `c1 = 1.0`, `c2 = 2.0`).

    **Example Changes (in `pso.py`):**

    ```python
    # Change swarm size:
    swarm_size = 20  # Originally 10

    # Change confidence levels:
    c1 = 1.0  # Originally 1.49
    c2 = 2.0  # Originally 1.49
    ```

9.  Write a program to minimize the function (x-y+7) where the minimum value x, or y can take should be -100, and the maximum value x, or y can take should be 100. Use a swarm size of 10, number of generations 2000, inertia weight `w = 0.729`, `c1 = 1.49`, and `c2 = 1.49`. (note: The minimum is: -193)

    **Answer code (pso.py):**

    ```python
    # -*- coding: utf-8 -*-

    """
    Created on Tue Mar 11 10:30:46 2025
    @author: umair
    """

    import random

    # PSO parameters
    swarm_size = 10
    num_generations = 100
    w = 0.729  # Inertia weight
    c1 = 1.49  # Cognitive coefficient
    c2 = 1.49  # Social coefficient
    min_boundary = -100
    max_boundary = 100

    # Function to minimize
    def fitness_function(x, y):
        return x - y + 7

    class Particle:
        def __init__(self):
            self.position = [random.uniform(min_boundary, max_boundary) for _ in range(2)]
            self.velocity = [random.uniform(-1, 1) for _ in range(2)]
            self.fitness = fitness_function(self.position[0], self.position[1])
            self.best_position = list(self.position)
            self.best_fitness = self.fitness

        def update_velocity(self, global_best_position):
            for d in range(2):
                r1, r2 = random.random(), random.random()
                cognitive_component = c1 * r1 * (self.best_position[d] - self.position[d])
                social_component = c2 * r2 * (global_best_position[d] - self.position[d])
                self.velocity[d] = w * self.velocity[d] + cognitive_component + social_component

        def update_position(self):
            for d in range(2):
                self.position[d] += self.velocity[d]
                # Boundary constraints
                self.position[d] = max(min(self.position[d], max_boundary), min_boundary)
            self.fitness = fitness_function(self.position[0], self.position[1])

    # Initialize swarm
    swarm = [Particle() for _ in range(swarm_size)]

    # Initialize global best
    global_best_position = min(swarm, key=lambda p: p.fitness).position
    global_best_fitness = fitness_function(global_best_position[0], global_best_position[1])

    # Main PSO loop
    for generation in range(num_generations):
        for particle in swarm:
            particle.update_velocity(global_best_position)
            particle.update_position()
            # Update personal best
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = list(particle.position)
            # Update global best
            if particle.fitness < global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_position = list(particle.position)
        # Print every 100 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {global_best_fitness}")

    # Final results
    print("Optimization Complete!")
    print(f"Best Position: {global_best_position}")
    print(f"Best Fitness Value: {global_best_fitness}")
    ```

    **Answer code2 (pso_animation.py):**

    ```python
    # -*- coding: utf-8 -*-

    """
    Created on Tue Mar 11 10:37:54 2025
    @author: umair
    """

    import random
    import numpy as np
    import matplotlib.pyplot as plt

    # PSO parameters
    swarm_size = 10
    num_generations = 50
    w = 0.729  # Inertia weight
    c1 = 1.49  # Cognitive coefficient
    c2 = 1.49  # Social coefficient
    min_boundary = -100
    max_boundary = 100

    # Function to minimize
    def fitness_function(x, y):
        return x - y + 7

    class Particle:
        def __init__(self):
            self.position = [random.uniform(min_boundary, max_boundary) for _ in range(2)]
            self.velocity = [random.uniform(-1, 1) for _ in range(2)]
            self.fitness = fitness_function(self.position[0], self.position[1])
            self.best_position = list(self.position)
            self.best_fitness = self.fitness

        def update_velocity(self, global_best_position):
            for d in range(2):
                r1, r2 = random.random(), random.random()
                cognitive_component = c1 * r1 * (self.best_position[d] - self.position[d])
                social_component = c2 * r2 * (global_best_position[d] - self.position[d])
                self.velocity[d] = w * self.velocity[d] + cognitive_component + social_component

        def update_position(self):
            for d in range(2):
                self.position[d] += self.velocity[d]
                # Boundary constraints
                self.position[d] = max(min(self.position[d], max_boundary), min_boundary)
            self.fitness = fitness_function(self.position[0], self.position[1])

    # Initialize swarm
    swarm = [Particle() for _ in range(swarm_size)]

    # Initialize global best
    global_best_position = max(swarm, key=lambda p: p.fitness).position
    global_best_fitness = fitness_function(global_best_position[0], global_best_position[1])

    # Visualization setup
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(min_boundary, max_boundary)
    ax.set_ylim(min_boundary, max_boundary)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("PSO Optimization of f(x,y) = x - y + 7")
    # Generate contour plot for visualization
    X, Y = np.meshgrid(np.linspace(min_boundary, max_boundary, 100), np.linspace(min_boundary, max_boundary, 100))
    Z = X - Y + 7
    ax.contourf(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.5)

    # Plot particles
    positions = np.array([p.position for p in swarm])
    scat = ax.scatter(positions[:, 0], positions[:, 1], c='red', label="Particles", edgecolors="black")
    global_best_marker, = ax.plot([], [], 'bo', markersize=10, label="Global Best")
    plt.legend()
    plt.ion()  # Enable interactive mode

    # Main PSO loop
    for generation in range(num_generations):
        for particle in swarm:
            particle.update_velocity(global_best_position)
            particle.update_position()
            # Update personal best
            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = list(particle.position)
            # Update global best
            if particle.fitness > global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_position = list(particle.position)
        # Update visualization
        positions = np.array([p.position for p in swarm])
        scat.set_offsets(positions)
        if isinstance(global_best_position, list) and len(global_best_position) == 2:
            global_best_marker.set_data([global_best_position[0]], [global_best_position[1]])
        ax.set_title(f"Iteration {generation + 1}")
        plt.draw()
        plt.pause(1)
    plt.ioff()  # Turn off interactive mode
    plt.show()

    # Final results
    print("Optimization Complete!")
    print(f"Best Position: {global_best_position}")
    print(f"Best Fitness Value: {global_best_fitness}")
    ```