import sys
import os
from typing import List

# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

from micrograd.engine import Value


class Metrics:

    def __init__(self, metrics: List[str] = []):
        # check that metrics list consist of only valid metrics [loss, accuracy, precision, recall, f1]
        for metric in metrics:
            assert metric in ["loss", "accuracy"], f"Invalid metric {metric}"

        self.metrics = metrics
        self.loss_history = {}
        self.output_history = {}

    def record(self, loss: float, preds: List[Value], y: List[Value], epoch: int, iteration: int):
        if iteration == 0:
            self.loss_history[f"Epoch {epoch + 1}"] = []
            self.output_history[f"Epoch {epoch + 1}"] = []

        self.loss_history[f"Epoch {epoch + 1}"].append(loss.data)
        predicted_probs = [p.data for p in preds]
        predicted_class = max(range(len(predicted_probs)), key=lambda i: predicted_probs[i])
        labels_class = [l.data for l in y]
        labels_class = labels_class.index(1)
        self.output_history[f"Epoch {epoch + 1}"].append([predicted_class, labels_class])

    def report(self, epoch: int, total_epochs: int):
        output = f"Epoch: {epoch + 1}/{total_epochs}"
        for metric in self.metrics:
            output += f" | {metric.capitalize()}: {self.calculate_metric_by_epoch(metric, epoch + 1):.4f}"
        output += "\n"
        output += "-" * len(output)
        print(output)
        

    def calculate_metric_by_epoch(self, metric: str, epoch: int) -> float:
        if metric == "loss":
            return self.calculate_mean_loss_by_epoch(epoch)
        elif metric == "accuracy":
            return self.calculate_multiclass_accuracy_by_epoch(epoch)
        return None

    def calculate_mean_loss_by_epoch(self, epoch: int) -> float:
        key = f"Epoch {epoch}"
        losses = self.loss_history[key]
        return sum(losses) / len(losses)
    
    def calculate_multiclass_accuracy_by_epoch(self, epoch: int) -> float:
        key = f"Epoch {epoch}"
        results = self.output_history[key]
        correct = 0
        for result in results:
            pred = result[0]
            y = result[1]
            if pred == y:
                correct += 1
        return correct / len(results)
    