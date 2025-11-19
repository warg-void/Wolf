# Wolf — C++ Neural Net Library

Wolf is a barebones C++23 neural network implementation.
Trains a fully connected network to **~95% accuracy on MNIST** on 1 epoch and CPU in **<1 second** without any external ML frameworks. It is also optimized for CPU and has been tested to run more than **5x** faster than pytorch's CPU.

# [Documentation](https://warg-void.github.io/wolf-docs/)

[![Build](https://github.com/warg-void/Wolf/actions/workflows/cmake-multi-platform.yml/badge.svg?branch=main)](https://github.com/warg-void/Wolf/actions/workflows/cmake-multi-platform.yml)
![C++](https://img.shields.io/badge/C%2B%2B-23-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Features 

- Fully connected feed-forward neural networks
- Backpropagation + Stochastic Gradient Descent
- Linear and ReLU Layers
- Fully optimized for CPU
- Batched Learning
- Helper functions to save and load neural nets
---

### Requirements

Recent C++ Compiler (gcc > 14)

### 1. Clone and build

```bash
git clone https://github.com/warg-void/Wolf.git
cd Wolf
cmake -B build -DBUILD_MNIST ON
cmake --build build
```
### 2. Run the examples

```bash
./build/examples/xortest
./build/examples/irisClassifier
./build/examples/mnistClassifier
```
## External Libraries Used
- OpenMP (for parallelization)
- [zpp_bits](https://github.com/eyalz800/zpp_bits) (for serializing and deserializing)

<img src="public/img/SWNO1.jpg" alt="Silver Wolf" width="400"/>

### To implement
- [ ] Adam and AdamW optimizers
- [ ] More Types of Layers and Loss
- [ ] Write Docs
- [ ] ROCm and rocBLAS 
- [ ] cuDA and cuBLAS
- [ ] CNNs
