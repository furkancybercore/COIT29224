# Models Directory

This directory is used to store trained models from the Neural Network with PSO project.

## Saved Models

When you train a model and use the `--output` option, the trained model will be saved in this directory. 

Models are stored in the Python pickle format (.pkl) and contain the neural network weights, configuration, and training history.

## Usage

To load a saved model, use the appropriate option in the main script:

```
python src/main.py --load models/my_model.pkl
``` 