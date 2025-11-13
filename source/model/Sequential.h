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

    Tensor forward(const Tensor& x) {
        Tensor out = x;
        for (auto& layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }

    Tensor backward(const Tensor& grad_out) {
        Tensor g = grad_out;
        for (size_t i = layers.size() - 1; i < layers.size(); --i) {
            // g.print();
            g = layers[i]->backward(g);
        }
        return g;
    }

    void step() { // SGD

    }
    Tensor pred(const Tensor &x);

private:
    std::vector<std::unique_ptr<Layer>> layers;
};

}
