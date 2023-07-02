# CNN from scratch

## Usage
    
Check `requirements.txt` for required packages.
```bash
python3 micrograd/main.py
```

## Code Structure

1. `micrograd/`: micrograd library and final project code
    - `main.py`: main file for running project
    - `config.py`: configuration file
    - `training.py`: training and testing loop
    - `engine.py`: engine for micrograd
    - `nn.py`: definition of nn models and layers
    - `optimizers.py`: definition of optimizers
    - `metrics.py`: definition of metrics

2. `test/`: files used for testing and debugging (not important)
    - `debugging.ipynb`: notebook for debugging
    - `test_linear.py`: test of linear layer (comparision with pytorch)
    - `test_linear_traning.py`: test of linear layer (training)
    - `test_binary_classification.py`: test of binary and multi-class classification
    - `test_engine.py`: test of engine (karpaty's test)

