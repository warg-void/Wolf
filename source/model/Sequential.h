#pragma once
#include <vector>
#include <memory>
#include <model/Layer.h>
namespace wolf {

class Sequential {
public:
    Sequential() = default;

    template<typename... LayerPtrs>
    Sequential(LayerPtrs&&... input_layers) {
        (layers.emplace_back(std::move(input_layers)), ...);
    }

    Tensor pred(const Tensor &x);
    Tensor backward(const Tensor& grad_out);
    void step(float lr);

private:
    std::vector<std::unique_ptr<Layer>> layers;
};

}
