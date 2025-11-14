#include <model/Sequential.h>

namespace wolf {
    Tensor Sequential::pred(const Tensor& x) {
        Tensor out = x;
        for (auto& l : layers) {
            out = l->forward(out);
        }
        return out;
    }

    Tensor Sequential::backward(const Tensor& grad_out) {
        Tensor g = grad_out;
        for (size_t i = layers.size() - 1; i < layers.size(); --i) {
            // g.print();
            g = layers[i]->backward(g);
        }
        return g;
    }
    
    void Sequential::step(float lr) { // learn rate
        for (size_t i = layers.size() - 1; i < layers.size(); --i) {
            layers[i]->step(lr);
        }
    }
}