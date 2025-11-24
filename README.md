# Wolf — A Fast, Native C++ Neural Net Library

Wolf is a barebones C++23 neural network implementation.
Trains a fully connected network at blazing fast speeds on CPU.

# [Documentation](https://warg-void.github.io/wolf-docs/)

[![Build](https://github.com/warg-void/Wolf/actions/workflows/cmake-multi-platform.yml/badge.svg?branch=main)](https://github.com/warg-void/Wolf/actions/workflows/cmake-multi-platform.yml)
![C++](https://img.shields.io/badge/C%2B%2B-23-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)


- [Wolf — A Fast, Native C++ Neural Net Library](#wolf--a-fast-native-c-neural-net-library)
- [Documentation](#documentation)
  - [Features](#features)
    - [Build from Source (CMake)](#build-from-source-cmake)
  - [External Libraries Used (No need to install)](#external-libraries-used-no-need-to-install)
    - [References](#references)
---
## Features 

- Fully connected feed-forward neural networks
- Backpropagation + Stochastic Gradient Descent
- Linear and ReLU Layers
- Fully optimized for CPU
- Batched Learning
- Helper functions to save and load neural nets
- Adam, Momentum and RMSProp Optimizer
- MSE, Cross Entropy and Binary Cross Entropy cross 
---

### Build from Source (CMake)

Requirements: Recent C++ Compiler (gcc > 14)

1. Clone and build

```bash
git clone https://github.com/warg-void/Wolf.git
cd Wolf
cmake -B build -DBUILD_MNIST ON
cmake --build build
```
2. Run the examples

```bash
./build/examples/xortest
./build/examples/irisClassifier
./build/examples/mnistClassifier
```
## External Libraries Used (No need to install)
- OpenMP (for parallelization)
- [zpp_bits](https://github.com/eyalz800/zpp_bits) (for serializing and deserializing)


<img src="public/img/SWNO1.jpg" alt="Silver Wolf" width="400"/>


### References
1) Deep Learning: Foundations and Concepts: by Christopher M. Bishop and Hugh Bishop, Springer Cham (2023).
