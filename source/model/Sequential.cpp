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
        int batch_size = grad_out.nrows();
        for (int i = 0; i < batch_size; i++) {
            auto &temp = grad_out.raw();
            auto it_begin = temp.begin() + i * grad_out.ncols();
            auto it_end = it_begin + grad_out.ncols();
            Tensor g = Tensor(std::vector<float>(it_begin, it_end), 1, grad_out.ncols());
            for (size_t i = layers.size() - 1; i < layers.size(); --i) {
                g = layers[i]->backward(g, batch_size); 
            }
        }
        return g;
    }
    
    void Sequential::step(float lr) { // learn rate
        for (size_t i = layers.size() - 1; i < layers.size(); --i) {
            layers[i]->step(lr);
        }
    }
}