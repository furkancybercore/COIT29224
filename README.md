# Neural Network with PSO Optimization

This project implements a Neural Network model optimized using Particle Swarm Optimization (PSO) for solving classification or prediction problems. The solution combines the learning capabilities of neural networks with the efficient search properties of PSO to achieve better performance compared to traditional training methods.

## Features

- Neural Network Implementation: A fully-connected neural network with configurable architecture
- PSO Integration: Implementation of Particle Swarm Optimization algorithm for training
- Data Processing: Tools for loading, preprocessing, and splitting datasets
- Performance Evaluation: Metrics calculation and visualization of results
- Model Export/Import: Save and load trained models for future use

## Installation

1. Clone the repository:
```
git clone https://github.com/furkancybercore/COIT29224.git
cd COIT29224
```

2. Create a virtual environment (recommended):
```
python -m venv pso_nn_env
# On Windows
pso_nn_env\Scripts\activate
# On macOS/Linux
source pso_nn_env/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the main script with your dataset:
```
python src/main.py --dataset path/to/dataset.csv
```

For more options:
```
python src/main.py --help
```

## Project Structure

- `src/`: Source code for the neural network and PSO implementation
- `data/`: Sample datasets and data handling utilities
- `models/`: Saved model files
- `tests/`: Unit tests

## License

This project is part of COIT29224 Assessment.