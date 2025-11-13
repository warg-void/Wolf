#pragma once
#include <tensor.h>
#include <model/Layer.h>

namespace wolf {

class LinearLayer : public Layer {
public:
    LinearLayer(size_t in_dim, size_t out_dim);


    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    // void zero_grad() override;

private:
    size_t in_dim;
    size_t out_dim;

    Tensor W;   // [out_dim x in_dim]
    Tensor dW;

    Tensor b;   // [out_dim x 1]
    Tensor db;

    Tensor last_input; // [B x in_dim]
};

} // namespace nn
