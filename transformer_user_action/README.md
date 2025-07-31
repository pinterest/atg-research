# TransAct 
## Transformer-based Realtime User Action Model for Recommendation at Pinterest

TransAct is a PyTorch module for modeling user sequences using the Transformer architecture. It is designed to handle user behavior data with sequential interactions, such as recommendation systems or user activity prediction.

## Paper
## TransAct v2
Xue Xia, Saurabh Vishwas Joshi, Kousik Rajesh, 
Kangnan Li, Yangyi Lu, Nikil Pancha,
Dhruvil Deven Badani, Jiajing Xu, Pong Eksombatchai . 2025.
TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation.

## TransAct
Xue Xia, Chantat Eksombatchai, Nikil Pancha, Dhruvil Deven Badani, Po-
Wei Wang,, Neng Gu, Saurabh Vishwas Joshi, Nazanin Farahpour, Zhiyuan
Zhang, Andrew Zhai . 2023. TransAct: Transformer-based Realtime User
Action Model for Recommendation at Pinterest.

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