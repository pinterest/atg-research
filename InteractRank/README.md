# InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-EE4C2C.svg)](https://pytorch.org/)

Official implementation of the InteractRank model from the WWW '25 paper: "InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features"

## Overview

InteractRank is a novel two-tower pre-ranking model that incorporates cross interaction features while maintaining computational efficiency. Key features include:

- Efficient integration of query-item cross interaction features in a two-tower architecture
- Historical user engagement-based interaction modeling
- Real-time user sequence modeling capabilities
- Significant performance improvements over BM25 and vanilla two-tower baselines

## Model Architecture

![InteractRank Architecture](assets/lw_scoring_v6.jpg)

InteractRank extends the traditional two-tower architecture by incorporating:

1. Query Tower: Processes search query inputs
2. Item Tower: Processes item features
3. Cross Interaction Layer: Captures historical user engagement patterns
4. Fusion Layer: Combines tower outputs with interaction features

The model achieves improved ranking quality while maintaining the computational efficiency needed for web-scale deployment.

## Installation

```bash
git clone https://github.com/pinterest/interactrank
cd interactrank
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from interactrank import InteractRankModel
from interactrank.data import SyntheticDataGenerator

# Initialize model
model = InteractRankModel(
    query_tower_dim=128,
    item_tower_dim=128,
    interaction_dim=64
)

# Load synthetic data for testing
train_data = SyntheticDataGenerator.generate_dataset(num_samples=1000)

# Train model
model.train(train_data)

# Get pre-ranking scores
scores = model.predict(queries, items)
```

### Training on Custom Data

See `examples/custom_training.py` for detailed examples of training on your own dataset.

## Baselines

This repository includes implementations of baseline models for comparison:

- BM25 (`baselines/bm25.py`)
- Vanilla Two-Tower Model (`baselines/two_tower.py`)

## Synthetic Dataset

The repository includes a synthetic dataset generator (`data/synthetic.py`) that creates sample data matching the expected input format. This can be used to:

- Understand the expected data format
- Test model implementation
- Run initial experiments

## Citation

If you use InteractRank in your research, please cite:

```bibtex
@inproceedings{khandagale2025interactrank,
  title={InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features},
  author={Khandagale, Sujay and Juneja, Bhawna and Agarwal, Prabhat and Subramanian, Aditya and Yang, Jaewon and Wang, Yuting},
  booktitle={Companion Proceedings of the ACM Web Conference 2025},
  year={2025},
  doi={10.1145/3701716.3715239}
}
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or paper, please contact Sujay Khandagale (skhandagale@pinterest.com).

## Acknowledgments

This research was conducted at Pinterest. We thank the Pinterest Search team for their support and feedback.