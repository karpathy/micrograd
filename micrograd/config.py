experiment_name = "mlp_classification"

# running configuration
run_training = True
run_testing = False

# model loading
load_model = False
load_path = "./experiments/mlp_classification/mlp_classification-2023-07-02_23:20:24.txt.pkl"

# model parameters
input_size = 5
hidden_size = [10, 10]
output_size = 3

# loss function [bce, ce]
loss = "ce"

# training parameters
epochs = 10
learning_rate = 0.03

# choose optimizer from ["sgd", "adam"]
optimizer = "sgd"

# metrics to track [loss, accuracy]
metrics = ["loss", "accuracy"]