
import sys
import os
import numpy as np
import dill
from typing import List

# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

from micrograd.engine import Value
from micrograd.nn import Linear, Sigmoid, BinaryCrossEntropyLoss, Sequential, Module, Softmax, CrossEntropyLoss
from micrograd.optimizers import SGD

from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import make_classification


def calculate_mean_loss(loss_history: List[float], epoch: int) -> float:
    key = f"Epoch {epoch}"
    losses = loss_history[key]
    return sum(losses) / len(losses)

def calculate_accuracy(result_history: List[float], epoch: int) -> float:
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

def calculate_multiclass_accuracy(result_history: List[float], epoch: int) -> float:
    key = f"Epoch {epoch}"
    results = result_history[key]
    correct = 0
    for result in results:
        pred = result[0]
        y = result[1]
        if pred == y:
            correct += 1
    return correct / len(results)


def save_model(model: Module, path: str):
    with open(path, "wb") as f:
        dill.dump(model, f)
        print(f"Model saved to {path}")

def load_model(path: str) -> Module:
    with open(path, "rb") as f:
        model = dill.load(f)
        print(f"Model loaded from {path}")
        return model

def create_dataset():
    # Generate a linearly separable dataset
    X, y = make_classification(
        n_samples=500, 
        n_features=5, 
        n_informative=2, 
        n_redundant=0,
        n_classes=3, 
        n_clusters_per_class=1,
        random_state=42
    )

    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)

    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse=False)

    # Reshape the labels list into a 2D array
    y_2d = [[label] for label in y]

    # Fit and transform the labels into one-hot encoded vectors
    y = encoder.fit_transform(y_2d)

    # Print the dataset
    print("Features (X):")
    print(X[:5])
    print("Labels (y):")
    print(y[:5])
    return X, y


def train_linear(x: List[List[Value]], y: List[Value],
                 in_neurons: int, nonlin=False, use_bias=True, load: bool = False, 
                 epochs: int = 10, learning_rate: float = 0.01):
    # model = Linear(in_neurons, 1, nonlin=nonlin ,use_bias=use_bias)

    if not load:
        # model = Sequential(Linear(in_neurons, 10, nonlin=True ,use_bias=use_bias),
        #                 Linear(10, 1, nonlin=nonlin ,use_bias=use_bias),
        #                 Sigmoid())
        model = Sequential(Linear(in_neurons, 10, nonlin=True ,use_bias=use_bias),
                           Linear(10, 3, nonlin=nonlin ,use_bias=use_bias),
                           Softmax())
        # model = Linear(in_neurons, 3, nonlin=nonlin ,use_bias=use_bias)
        # activation = Softmax()
    else:
        model = load_model("checkpoints/binary_classification/linear_model.pkl")
    # activation = Sigmoid()
    print(f"Number of training parameters: {len(model.parameters())}")
        
    # criterion = BinaryCrossEntropyLoss()
    criterion = CrossEntropyLoss(3)
    optimizer = SGD(lr=learning_rate)
    loss_history = {}
    result_history = {}

    for epoch in range(epochs):
        loss_history[f"Epoch {epoch + 1}"] = []
        result_history[f"Epoch {epoch + 1}"] = []
        for i, (x_, y_) in enumerate(zip(x, y)):
            pred = model(x_)
            # print(f"After model: {pred}")
            # pred = activation(pred)
            # print(f"After softmax: {pred}")
            # print(f"Labels: {y_}")
            # print(f"Linear: {pred[0].data}")
            # pred = activation(pred)
            # print(f"Sigmoid {pred[0].data}")
            loss = criterion(pred, y_)
            # print(f"Loss: {loss.data}")
            loss_history[f"Epoch {epoch + 1}"].append(loss.data)
            predicted_probs = [p.data for p in pred]
            predicted_class = max(range(len(predicted_probs)), key=lambda i: predicted_probs[i])
            labels_class = [l.data for l in y_]
            labels_class = labels_class.index(1)
            result_history[f"Epoch {epoch + 1}"].append([predicted_class, labels_class])
            # result_history[f"Epoch {epoch + 1}"].append([pred[0].data, y_[0].data])
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
    save_model(model, "checkpoints/binary_classification/linear_model.pkl")


def inference(x: List[List[Value]], y: List[Value], model: Module, metrics: List[str] = []):
    pass
    


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
        temp = []
        for val in label:
            temp.append(Value(val))
        train_labels.append(temp)
    print(f"Number of instances {len(train_data)}")
    # Train a linear model
    train_linear(train_data, train_labels, in_neurons=10, nonlin=False, load=False, epochs=30, learning_rate=0.03)

