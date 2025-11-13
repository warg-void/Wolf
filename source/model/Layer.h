#pragma once
#include <math/tensor.h>
#include <vector>
#include <memory>

namespace wolf {

class Layer {
public:
    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& grad_out) = 0; // input: gradient of the output, output: gradient of the input 
    virtual ~Layer() = default;
    // virtual void zero_grad() {}
};

}
