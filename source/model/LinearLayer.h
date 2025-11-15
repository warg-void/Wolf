#pragma once
#include <math/tensor.h>
#include <model/Layer.h>

namespace wolf {

class LinearLayer : public Layer {
public:
    LinearLayer(size_t in_dim, size_t out_dim);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out, int batch_size) override;
    void step(float lr);
private:
    size_t in_dim;
    size_t out_dim;

    std::vector<size_t> idx; // [out_dim x in_dim] for indexing

    Tensor W;   // [out_dim x in_dim]
    Tensor dW;

    Tensor b;   // [out_dim x 1]
    Tensor db;

    Tensor last_input; // [B x in_dim]
};

} // namespace nn
