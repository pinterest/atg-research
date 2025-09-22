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

## Installation and Usage

```bash
git clone https://github.com/pinterest/atg-research.git
cd InteractRank/interactrank
pip install -r requirements.txt
chmod 777 run.sh
./run.sh -> this will trigger the config bundle for training LW Model with synthetic data for one batch
```

```

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

For question about the code or paper, please contact Bhawna Juneja (bjuneja@pinterest.com).

## Acknowledgments

This research was conducted at Pinterest. We thank the Pinterest Search team for their support and feedback.
