# PyTorch LMU
This repository contains PyTorch implementations of the following papers: 
- [Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks](https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks), by Voelker AR, Kajić I, and Eliasmith C
- [Parallelizing Legendre Memory Unit Training](https://arxiv.org/abs/2102.11417), by Chilkuri N and Eliasmith C  
 
Performance on the psMNIST dataset is demonstrated in [`examples/`](examples).

## Usage
`torch`, `numpy`, and `scipy` are the only requirements.  
[`src/lmu.py`](src/lmu.py) contains the implementations of `LMUCell`, `LMU` and `LMUFFT`.  
  
**Examples:**

- LMU
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

- LMUFFT
    ```python3
    import torch
    from lmu import LMUFFT

    model = LMUFFT(
        input_size = 1,
        hidden_size = 346,
        memory_size = 468, 
        seq_len = 784, 
        theta = 784
    )

    x = torch.rand(100, 784, 1) # [batch_size, seq_len, input_size]
    output, h_n = model(x)
    ```

## Running on psMNIST
- Clone this repository and open: 
  - [`examples/lmu_psmnist.ipynb`](examples/lmu_psmnist.ipynb), for training and evaluating an LMU model on the psMNIST dataset
  - [`examples/lmu_fft_psmnist.ipynb`](examples/lmu_fft_psmnist.ipynb), for training and evaluating an LMUFFT model on the psMNIST dataset  
  
  Running in [Google Colab](https://colab.research.google.com/) is preferred  
- [`examples/permutation.pt`](examples/permutation.pt) contains the permutation tensor used while creating the psMNIST data; it's included for reproducibility. Alternatively, `torch.randperm(784)` can be used to test with a new permutation.  

## References
- [Voelker, Aaron R., Ivana Kajić, and Chris Eliasmith. "Legendre memory units: Continuous-time representation in recurrent neural networks." (2019).](https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks)
- [Chilkuri, Narsimha, and Chris Eliasmith. "Parallelizing Legendre Memory Unit Training." (2021)](https://arxiv.org/abs/2102.11417)
- Official Keras implementation of LMU: [nengo/keras-lmu](https://github.com/nengo/keras-lmu)
- Official Keras implementation of LMUFFT: [NarsimhaChilkuri/Parallelizing-Legendre-Memory-Unit-Training](https://github.com/NarsimhaChilkuri/Parallelizing-Legendre-Memory-Unit-Training)
- [Legendre Memory Units in NengoDL](https://www.nengo.ai/nengo-dl/examples/lmu.html)
