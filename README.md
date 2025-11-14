# Wolf — C++ Neural Net Library

Wolf is a no-dependency C++23 neural network implementation with a clean CMake setup and runnable examples.  
Trains a fully connected network to **~96% accuracy on MNIST** in 3 epochs without any external ML frameworks.

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![C++](https://img.shields.io/badge/C%2B%2B-23-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Implemented Already

- Fully connected feed-forward neural networks (MLPs)
- Backpropagation + Stochastic Gradient Descent
- Linear and ReLU Layers

---

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

<img src="public/img/SWNO1.jpg" alt="Silver Wolf" width="400"/>

### To implement
- [ ] Batched learning
- [ ] Adam and AdamW optimizers
- [ ] Helper functions to save and load neural nets
- [ ] Helper functions to download and parse datasets
- [ ] Write Docs
- [ ] ROCm and rocBLAS 
- [ ] cuDA and cuBLAS
- [ ] CNNs
- [ ] CPU matrix optimizations