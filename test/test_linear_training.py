import sys
import os
import numpy as np
from typing import List

# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

from micrograd.engine import Value
from micrograd.nn import Linear, CrossEntropyLoss
from micrograd.optimizers import SGD

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


def create_dataset():
    # Load the Iris dataset
    iris = load_iris()

    # Extract the features (X) and target labels (y)
    X = iris.data
    y = iris.target.reshape(-1, 1)  # Reshape to column vector

    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)

    # Convert the class labels to one-hot encodings
    encoder = OneHotEncoder()
    y_one_hot = encoder.fit_transform(y).toarray()

    # Print the first 5 samples
    print("X:", X[:5])
    print("y (one-hot):", y_one_hot[:5])

    return X, y_one_hot


def train_linear(x: List[List[Value]], y: List[Value], num_classes: int,
                 in_neurons: int, out_neurons: int, nonlin=False, use_bias=True, 
                 epochs: int = 10):
    model = Linear(in_neurons, out_neurons, nonlin=nonlin ,use_bias=use_bias)
    print(f"Number of training parameters: {len(model.parameters())}")    
    criterion = CrossEntropyLoss(num_classes)
    optimizer = SGD()

    print(f"Number of training parameters: {len(model.parameters())}")
    for i, (x_, y_) in enumerate(zip(x, y)):
        #if i > 5 : break
        pred = model(x_)
        # print(pred)
        # print(y_)
        loss = criterion(pred, y_)
        print(f"Loss: {loss.data:.4f}")

        # backward
        loss.backward()
        optimizer.step(model.parameters())

        # zero gradients
        model.zero_grad()
        loss.destroy_graph(model.parameters())

    # for epoch in range(epochs):
    #     # forward
    #     for x_, y_ in zip(x, y):
    #         pred = model(x_)
    #         print(pred)
    #         print(y_)
    #         loss = criterion(pred, y_)
    #         print(f"Loss: {loss.data:.4f}")

    #         # backward
    #         loss.backward()
    #         optimizer.step(model.parameters())

    #         # zero gradients
    #         model.zero_grad()

    #     print(f"Epoch: {epoch + 1}/{epochs} | Loss: {loss.data:.4f}")

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
    X = train_data
    for label in y:
        temp = []
        for val in label:
            temp.append(Value(val))
        train_labels.append(temp)
    y = train_labels

    # Train a linear model
    train_linear(X, y, 3, in_neurons=4, out_neurons=3, nonlin=True, epochs=5)
