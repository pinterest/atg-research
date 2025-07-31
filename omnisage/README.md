# omnisage-os

A minimal, open-source-friendly version of an internal project for graph neural network (GNN) training, supporting metric learning, user sequence modeling, and autoencoding tasks. The paper is available at https://arxiv.org/abs/2504.17811

## Main Files

- **model.py**: Contains the OmniSage model, including feature handling, neighbor aggregation, and embedding logic.
- **tasks.py**: Implements the main training tasks: metric learning, user sequence modeling, and autoencoder, each with a unified softmax cross entropy loss.
- **train.py**: The main training script. Generates synthetic data, runs multi-task training, and prints losses for each task.

## How to Run

Install dependencies (see requirements.txt), then run:

```bash
python train.py
```

This will launch a synthetic multi-task training loop for all supported objectives. 
