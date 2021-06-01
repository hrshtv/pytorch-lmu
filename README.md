# PyTorch LMU
This repository contains a PyTorch implementation of Legendre Memory Units (LMUs), as presented in the [NeurIPS 2019 paper](https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks) by Voelker AR, Kajić I, and Eliasmith C.  
SOTA performance on the psMNIST dataset is reproduced in [`examples/`](examples).

## Usage
`torch`, `numpy`, and `scipy` are the only requirements.  
[`src/lmu.py`](src/lmu.py) contains the implementations of `LMUCell` and `LMU`.  
A quick example:  
```python3
import torch
from lmu import LMU

model = LMU(
    input_size = 1,
    hidden_size = 212,
    memory_size = 256,
    theta = 784
)

x = torch.rand(100, 784, 1) # [batch_size, seq_len, input_size]
output, (h_n, m_n) = model(x)
```

## Running on psMNIST
- Clone this repository and open [`examples/lmu_psmnist.ipynb`](examples/lmu_psmnist.ipynb) (running in [Google Colab](https://colab.research.google.com/) is preferred)
- [`examples/permutation.pt`](examples/permutation.pt) contains the permutation tensor used while creating the psMNIST data; it's included for reproducibility. Alternatively, `torch.randperm(784)` can be used to test with a new permutation.  

## References
- [Voelker, Aaron R., Ivana Kajić, and Chris Eliasmith. "Legendre memory units: Continuous-time representation in recurrent neural networks." (2019).](https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks)
- [Legendre Memory Units in NengoDL](https://www.nengo.ai/nengo-dl/examples/lmu.html)
- [nengo/keras-lmu](https://github.com/nengo/keras-lmu)
