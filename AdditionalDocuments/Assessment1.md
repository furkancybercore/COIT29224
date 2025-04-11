# COIT29224	Evolutionary	Computation

Assignment	Item	1,	Term 1,	2025

Due	Date  
Friday of	Week 5	(11 April 2025)	11.00 pm	AEST

Weighting  
25%

Assessment	Task  
Enhancing	Neural	Network	Performance	with	Particle	Swarm	
Optimisation (PSO)

## Objectives

The	purpose	of	this	assessment	item	is	to	assess	your	skills	attributable	to	the	following	learning outcomes and your achievement of the expected graduate attributes of advanced level communication, cognitive, technical, and creative skills, self-management, and professional-level knowledge.

1. Formulate an evolutionary computation search or optimization problem by analysing an authentic case or scenario.
2. Design an evolutionary algorithm for a problem applying the core evolutionary computation concepts and mechanisms.
3. Build a software application to implement an evolutionary algorithm for a complex search or optimization problem.

**Assessment Task**

Assessment 1 is an individual task. Its primary objective is to design, code and build software application using PSO using the topics learnt in week 1-5. You need to analyse Particle Swarm Optmisation (PSO) define a problem specification and build a optimisation solution using Particle Swarm Optmisation (PSO). Additionally, you should compile a report containing the Python code, screenshots of the output that demonstrate the solutions to the questions, and an evaluation of the produced results.

Be aware that every report for Assessment 1 is run through an automated plagiarism detection system. The teaching team can effortlessly spot instances of copied or plagiarised content.

- Plagiarism may result in consequences varying from mark reductions to failing the course, or even removal from the University.
- Make sure you're acquainted with the Academic Misconduct Procedures, which can be found at:  
  http://policy.cqu.edu.au/Policy/policy_file.do?policyid=1244.

## Section A: Problem Statement – Enhancing Neural Network Performance with Particle Swarm Optimization

Recent advancements in machine learning have spotlighted the critical role of hyperparameter optimization in enhancing the performance of neural networks across various domains. Traditional optimization methods often involve manual selection or grid/random search, which can be time-consuming and may not always yield the best results. Particle Swarm Optimization (PSO), a bio-inspired algorithm, has emerged as a promising alternative, capable of efficiently navigating the search space to find optimal or near-optimal solutions. This assessment you have to apply PSO for optimizing the hyperparameters of neural networks (NNs) or Convolutional Neural Networks (CNNs), and to demonstrate the superiority of this approach over traditional optimization methods in a chosen application area.

Students will select a problem domain of their interest where neural networks are applicable. They will then design and implement a PSO-optimized neural network solution (PSO-NN) for this problem. The primary goal is to show that the PSO-NN model outperforms a traditionally optimized neural network model (e.g., using grid search, random search, or manual tuning) in terms of accuracy, efficiency, or other relevant performance metrics.

## Section B: Task Description

1. **Problem Selection:**
   - Choose a problem where neural networks are applicable. This could be in areas such as image recognition, natural language processing, time series prediction, or any other domain of interest.
   - Define clear objectives and success metrics for the chosen problem.
2. **Data Acquisition and Pre-processing:**
   - Collect or identify a suitable dataset for the problem.
   - Perform necessary pre-processing steps to prepare the data for training and testing the models.
3. **Traditional Neural Network Implementation:**
   - Design and implement a neural network or CNN model as a baseline solution to the chosen problem.
   - Optimize the model’s hyperparameters using traditional methods (e.g., grid search, random search, or manual tuning).
4. **PSO-Optimized Neural Network Implementation:**
   - Implement PSO to optimize the hyperparameters of the neural network model designed in the previous step.
   - Ensure that the PSO algorithm optimizes key hyperparameters that significantly impact model performance, such as learning rate, number of layers, number of neurons in each layer, activation functions, etc.
5. **Performance Comparison and Analysis:**
   - Compare the performance of the PSO-NN model with the traditionally optimized neural network model using the predefined success metrics.
   - Conduct statistical tests if necessary to prove the significance of the results.
6. **Discussion and Conclusion:**
   - Discuss the findings, highlighting the effectiveness of PSO in optimizing the neural network’s hyperparameters.
   - Reflect on the advantages and potential limitations of the PSO-NN approach.
   - Suggest future work or improvements that could be made to the PSO-NN model.

## Section C: Source Code

You can use the Jupiter notebook or python to implement the solution with meaningful comments in the code. Upload your code in GitHub and share the link in the report.

## Section D: Report

Write a report with screenshot and your learnings and evaluation. Also show an instruction how to run the code as an instruction for new users.

## Section E: Software Tools and Building the Application

You can build your application using Anaconda/PyCharm.

## Section F: Submission Instructions

You should submit one zip file containing the following files using the Moodle online submission system. (Note: the file names/class names could be changed to meaningful names.)

## Section G: Reference

Here are some examples of problem that utilise the concept of Neural Network with PSO:

- **Cardiovascular disease prediction with PSO-NN**  
  (https://www.kaggle.com/code/zzettrkalpakbal/cardiovascular-disease prediction-with-pso-nn)
- **Predicting secondary school student performance using a double particle swarm optimization-based categorical boosting model**  
  (https://www.sciencedirect.com/science/article/abs/pii/S0952197623008333)
- **Implementation of Particle Swarm Optimization (PSO) to improve neural network performance in univariate time series prediction**  
  (https://kinetik.umm.ac.id/index.php/kinetik/article/view/1330/124124283)
- **MLP-PSO Hybrid Algorithm for Heart Disease Prediction**  
  (https://www.mdpi.com/2075-4426/12/8/1208)
- **Optimizing a Convolutional Neural Network using Particle Swarm Optimization**  
  (https://ieeexplore.ieee.org/document/9847314)
- **Particle swarm optimization based artificial neural network (PSO-ANN) model for effective k-barrier count intrusion detection system in WSN**  
  (https://www.sciencedirect.com/science/article/pii/S266591742300211)
- **A Particle Swarm Optimization Based Deep Learning Model for Vehicle Classification**  
  (https://www.techscience.com/csse/v40n1/44231/html)

## Assessment Item 1 Marking criteria

| Serial No. | Specification | Marks | Marks Scored |
|------------|---------------|-------|--------------|
| 1 | Originality and Relevance of the Chosen Problem<br>- Creativity in problem selection<br>- Relevance and Applicability of neural networks of the chosen problem | 4 | |
| 2 | Implementation and Technical Depth<br>- Correctness of the neural network and PSO Implementation<br>- Depth of understanding of the chosen problem, neural network models and PSO algorithm<br>- Correct modification of particle positions<br>- Correct use of informants<br>- Correctly sized dimensions for particles and creation of random values for the position | 10 | |
| 3 | Performance Comparison and Statistical Analysis<br>- Performance comparison between PSO-NN and traditional NN<br>- Appropriateness and correctness of the statistical analysis methods used | 4 | |
| 4 | Discussion, Conclusion and Future work<br>- Summarise the findings<br>- Suggestions for future work or improvements | 2 | |
| 5 | Good coding practices<br>1. Use of Classes, Methods (1)<br>2. Modularise (1)<br>3. Comments clearly describing the code segments (1)<br>4. Naming Conventions (1) | 2 | |
| 6 | Well-presented report with student details, providing correct screenshot to proof the implementation<br>1. Table of contents and reference (1)<br>2. Explanation on the library chosen (1)<br>3. Screenshot of the output (1)<br>4. Explanation on the evaluation (1)<br>5. Explain the model selection (1) | 3 | |
| 8 | Late Penalty (5% of total marks per day – 1 mark) | | |
| 9 | Plagiarism (as per policy) | | |
| **Total** |  | **25** | |

**Note:**
1. If your program does not compile or run, partial marks will be allocated by inspection of the source code.
2. Your understanding of the algorithm and problem-solving approach used will be examined using the detailed comments (docstrings) inserted in your source code file.
3. Please clarify any doubts you have by one of the means of discussing with your tutor, posting a query in the Q & A forum, or discussing with your colleagues or contacting the unit coordinator.
4. Please do not share your source code files or report with your colleagues which may lead to plagiarism. Please do not search and use source code available elsewhere which may also lead to plagiarism.