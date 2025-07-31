# TransAct 

TransAct is a PyTorch-based module developed at Pinterest for modeling user sequences using the Transformer architecture, specifically for real-time recommendation scenarios. It is built to handle sequential user behavior data, making it well-suited for applications like recommendation systems and user activity prediction. This repository provides the implementation for both TransAct and TransAct v2. Both versions share the same model backbone, while TransAct v2 introduces additional Triton kernels to support efficient inference on lifelong user sequences.

## Papers
## TransAct v2: Lifelong User Action Sequence Modeling on Pinterest Recommendation
Xue Xia, Saurabh Vishwas Joshi, Kousik Rajesh, 
Kangnan Li, Yangyi Lu, Nikil Pancha,
Dhruvil Deven Badani, Jiajing Xu, Pong Eksombatchai . 2025.
TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation.
https://arxiv.org/abs/2506.02267

## TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest
Xue Xia, Chantat Eksombatchai, Nikil Pancha, Dhruvil Deven Badani, Po-
Wei Wang, Neng Gu, Saurabh Vishwas Joshi, Nazanin Farahpour, Zhiyuan
Zhang, Andrew Zhai. 2023. TransAct: Transformer-based Realtime User
Action Model for Recommendation at Pinterest.
https://arxiv.org/abs/2306.00248

## Requirements

- Python 3.9.7 or higher: You can download and install Python from the official Python website at [https://www.python.org/downloads/](https://www.python.org/downloads/) or by using a package manager like conda or pip.
- torch 2.1 or higher
- triton 3.0
   
## Usage
```
python transact_code/test_run_transact.py
```

## Triton kernels
Triton kernel for Single Kernel Unified Transformer (SKUT) is available in skut.py, to run the kernel on sample inputs and compare performance with torch use
```
python skut.py
```

Triton kernel for sparse nearest neighbor search in a batch sparse long input tensor is available in sparse_nn.py, to run the kernel on sample inputs and compare performance with torch use
```
python sparse_nn.py
```
