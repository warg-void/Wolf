#pragma once
#include <math/tensor.h>
#include <model/Layer.h>
#include <external/zpp_bits.h>

namespace wolf {

// Symbol meanings:
// y = Wx + b

class LinearLayer : public Layer {
public:
    LinearLayer(size_t x_dim, size_t y_dim);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    void step_SGD(float lr, size_t batch_size) override;
    void step_momentum(float lr, float mu, size_t batch_size) override;
    void step_RMSProp(float lr, float alpha, float eps, size_t batch_size) override;
    void step_Adam(float lr, float beta1, float beta2, float eps, float bc1, float bc2, size_t batch_size) override;
    size_t in_size() const {return x_dim;}
    size_t out_size() const {return y_dim;}
    Tensor weights() const {return W;}
    Tensor bias() const {return b;}
    void save_body(zpp::bits::out<std::vector<std::byte>>& out) const override {
        const auto& Wv = W.raw();
        const auto& bv = b.raw();
        out(x_dim, y_dim, Wv, bv).or_throw();
    }
    static std::unique_ptr<Layer> load_from(zpp::bits::in<std::vector<std::byte>>& in) {
        std::size_t x_dim{}, y_dim{};
        std::vector<float> Wv, bv;
        in(x_dim, y_dim, Wv, bv).or_throw();

        auto layer = std::make_unique<LinearLayer>(x_dim, y_dim);
        layer->W.raw() = std::move(Wv);
        layer->b.raw() = std::move(bv);
        return layer;
    }
private:
    size_t x_dim;
    size_t y_dim;
    Tensor W;   // [out_dim x in_dim]
    Tensor dW;
    Tensor b;   // [out_dim x 1]
    Tensor db;
    Tensor last_input; // [B x in_dim]
    Tensor vW; // Momentum term
    Tensor vb;
    Tensor rW; // RMSProp term
    Tensor rb;
    float mu;
};

} // namespace nn
