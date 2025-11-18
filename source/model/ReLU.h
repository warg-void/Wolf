// relu.h
#pragma once
#include <model/Layer.h>
#include <math/tensor.h>

namespace wolf {

class ReLULayer : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;

    void step(float lr, size_t batch_size) {}
    ReLULayer() {}

private:
    Tensor last_input;
};

}
