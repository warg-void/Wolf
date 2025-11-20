// relu.h
#pragma once
#include <model/Layer.h>
#include <math/tensor.h>

namespace wolf {

class ReLULayer : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;

    void step(float lr, size_t batch_size) override {}
    void step_SGD(float lr, size_t batch_size) override {}
    void step_momentum(float lr, float mu, size_t batch_size) override {}
    void step_RMSProp(float lr, float alpha, float eps, size_t batch_size) override {}
    void step_Adam(float lr, float beta1, float beta2, float eps, float bc1, float bc2, size_t batch_size) override {}
    void save_body(zpp::bits::out<std::vector<std::byte>>& out) const override {
        // nothing
        (void)out;
    }
    ReLULayer() : Layer(LayerKind::ReLU) {}
    static std::unique_ptr<Layer> load_from(zpp::bits::in<std::vector<std::byte>>& in) {
        return std::make_unique<ReLULayer>();
    }

private:
    Tensor last_input;
};

}
