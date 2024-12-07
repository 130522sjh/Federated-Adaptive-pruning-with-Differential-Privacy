# Federated-Adaptive-Pruning-with-Differential-Privacy

This project explores **Federated Adaptive Pruning with Differential Privacy** (FAP-DP) within the context of federated learning. The aim is to combine adaptive pruning techniques with differential privacy to improve model efficiency and ensure privacy protection during training. 

### Current Experiments:
So far, experiments have been conducted using the **SVHN** and **CIFAR-10** datasets, under both **IID** (Independent and Identically Distributed) and **non-IID** settings.

## 🚀 Overview

Federated learning allows clients to collaboratively train a model without sharing their data. This project proposes an adaptive pruning approach to reduce model size and enhance performance, coupled with differential privacy to preserve individual privacy.

Key features of this project:
- **Adaptive Pruning**: Dynamically reduces model size during training to increase efficiency.
- **Differential Privacy**: Ensures privacy by adding noise to the gradients, making it difficult to infer information about individual data points.
- **SVHN & CIFAR-10 experiments**: Evaluation on well-known benchmark datasets under various data distributions.



## 🛠️ Setup

Ensure that you have the following prerequisites installed before running the scripts:

### Requirements

- Python 3.7+
- PyTorch (>=1.8.0)
- torchvision
- numpy
- matplotlib
- other necessary libraries (listed in `requirements.txt`)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/130522sjh/Federated-Adaptive-pruning-with-Differential-Privacy.git

## 🚀 Running the Code
For example:
> python main_fed.py --dataset svhn --model cnn --epochs 50 --gpu 0 --batch_size 10 --learning_rate 0.01 --local_epochs 5 --all_clients

`--all_clients` for averaging over all client models


Configuring Parallel Computing (Optional)
The scripts might be slow without parallel computing. To speed up execution, you can implement parallelization methods such as:

Multi-threading or multi-processing to handle multiple clients simultaneously.
Leverage GPU acceleration for faster model training (make sure PyTorch is installed with CUDA support).
Implementing parallelism is highly recommended for large-scale federated learning tasks.

⚡ Performance Considerations
The current implementation is sequential, meaning that each client's training process happens one after another, which can result in slower execution times.
To improve performance, consider using parallel computing or distributed systems.
For GPU acceleration, ensure that your environment supports CUDA (for PyTorch) to speed up model training and reduce time consumption.
