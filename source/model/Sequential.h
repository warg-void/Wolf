#pragma once
#include <vector>
#include <memory>
#include <model/Layer.h>
#include <model/optimizers.h>

namespace wolf {

class Sequential {
public:
    Sequential() = default;

    template<typename... LayerPtrs>
    Sequential(LayerPtrs&&... input_layers) {
        (layers.emplace_back(std::move(input_layers)), ...);
    }
    void set_optimizer(OptimVariant cfg);
    Tensor pred(const Tensor &x);
    TensorView pred(TensorView x);
    Tensor backward(const Tensor& grad_out);
    TensorView backward();
    void step(float lr, size_t batch_size = 1);
    void step(size_t batch_size = 1);

    TensorView grad_loss(const TensorView& a, const TensorView& b);
    void save(const std::string &path) const;
    static Sequential load(const std::string &path);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    Tensor fbuf; // Forward Buffer
    Tensor bbuf; // Backward buffer
    Tensor grad_out; // dE_dy
    std::optional<OptimVariant> optim_cfg;
    size_t step_t = 0;
};

}
