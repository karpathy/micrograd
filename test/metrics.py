from typing import List


class Metrics:

    def __init__(self, metrics: List[str] = []):
        self.metrics = metrics
        self.loss_history = {}
        self.output_history = {}

    def calculate_mean_loss_by_epoch(self, epoch: int) -> float:
        key = f"Epoch {epoch}"
        losses = self.loss_history[key]
        return sum(losses) / len(losses)

        