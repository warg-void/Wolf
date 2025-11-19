#pragma once
#include <math/tensor.h>
#include <model/Layer.h>
#include <external/zpp_bits.h>

namespace wolf {

class LinearLayer : public Layer {
public:
    LinearLayer(size_t in_dim, size_t out_dim);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    void step(float lr, size_t batch_size) override;
    size_t in_size() const {return in_dim;}
    size_t out_size() const {return out_dim;}
    Tensor weights() const {return W;}
    Tensor bias() const {return b;}
    void save_body(zpp::bits::out<std::vector<std::byte>>& out) const override {
        const auto& Wv = W.raw();
        const auto& bv = b.raw();
        out(in_dim, out_dim, Wv, bv).or_throw();
    }
    static std::unique_ptr<Layer> load_from(zpp::bits::in<std::vector<std::byte>>& in) {
        std::size_t in_dim{}, out_dim{};
        std::vector<float> Wv, bv;
        in(in_dim, out_dim, Wv, bv).or_throw();

        auto layer = std::make_unique<LinearLayer>(in_dim, out_dim);
        layer->W.raw() = std::move(Wv);
        layer->b.raw() = std::move(bv);
        return layer;
    }
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
