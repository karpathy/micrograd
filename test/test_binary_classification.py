
import sys
import os
import numpy as np
from typing import List

# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

from micrograd.engine import Value
from micrograd.nn import Linear, Sigmoid, BinaryCrossEntropyLoss
from micrograd.optimizers import SGD

from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification


def calculate_mean_loss(loss_history: List[Value], epoch: int) -> float:
    key = f"Epoch {epoch}"
    losses = loss_history[key]
    return sum(losses) / len(losses)

def calculate_accuracy(result_history: List[Value], epoch: int) -> float:
    key = f"Epoch {epoch}"
    results = result_history[key]
    correct = 0
    for result in results:
        pred = result[0]
        y = result[1]
        if pred > 0.5 and y == 1:
            correct += 1
        elif pred < 0.5 and y == 0:
            correct += 1
    return correct / len(results)

def create_dataset():
    # Generate a linearly separable dataset
    X, y = make_classification(
        n_samples=100, 
        n_features=2, 
        n_informative=2, 
        n_redundant=0, 
        n_clusters_per_class=1,
        random_state=42
    )

    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)

    # Print the dataset
    print("Features (X):")
    print(X[:5])
    print("Labels (y):")
    print(y[:5])
    return X, y


def train_linear(x: List[List[Value]], y: List[Value],
                 in_neurons: int, nonlin=False, use_bias=True, 
                 epochs: int = 10, learning_rate: float = 0.01):
    model = Linear(in_neurons, 1, nonlin=nonlin ,use_bias=use_bias)
    activation = Sigmoid()
    print(f"Number of training parameters: {len(model.parameters())}")
        
    criterion = BinaryCrossEntropyLoss()
    optimizer = SGD(lr=learning_rate)
    loss_history = {}
    result_history = {}

    for epoch in range(epochs):
        loss_history[f"Epoch {epoch + 1}"] = []
        result_history[f"Epoch {epoch + 1}"] = []
        for i, (x_, y_) in enumerate(zip(x, y)):
            pred = model(x_)
            # print(f"Linear: {pred[0].data}")
            pred = activation(pred)
            # print(f"Sigmoid {pred[0].data}")
            loss = criterion(pred, y_)
            loss_history[f"Epoch {epoch + 1}"].append(loss.data)
            result_history[f"Epoch {epoch + 1}"].append([pred[0].data, y_[0].data])
            # print(f"Loss: {loss.data:.4f}")

            # backward
            loss.backward()
            optimizer.step(model.parameters())

            # zero gradients
            model.zero_grad()
            loss.destroy_graph(model.parameters())

        print(f"Epoch: {epoch + 1}/{epochs} | Loss: {calculate_mean_loss(loss_history, epoch + 1):.4f}" + 
              f"| Accuracy: {calculate_accuracy(result_history, epoch + 1):.4f}")
        print("--------------------------------------------------")


if __name__ == "__main__":
    # Create a dataset for classification
    X, y = create_dataset()

    # Convert the dataset to micrograd values
    train_data = []
    train_labels = []
    for x in X:
        temp = []
        for val in x:
            temp.append(Value(val))
        train_data.append(temp)
    train_labels = []
    for label in y:
        train_labels.append([Value(label)])
    print(f"Number of instances {len(train_data)}")
    # Train a linear model
    train_linear(train_data, train_labels, in_neurons=10, nonlin=False, epochs=10, learning_rate=0.003)

