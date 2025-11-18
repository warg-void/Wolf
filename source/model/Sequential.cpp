#include <model/Sequential.h>

namespace wolf {
    Tensor Sequential::pred(const Tensor& x) {
        Tensor out = x;
        for (auto& l : layers) {
            out = l->forward(out);
        }
        return out;
    }

    TensorView Sequential::pred(TensorView x) {

        if (fbuf.empty() || fbuf.nrows() < x.rows || fbuf.ncols() < x.cols) {
            fbuf = Tensor(std::vector<float>(x.rows * x.cols), x.rows, x.cols);
        } 
        std::copy_n(x.data, x.cols * x.rows, fbuf.raw().begin());
        fbuf.set_cols(x.cols);
        fbuf.set_rows(x.rows);
        
        for (auto& l : layers) {
            fbuf = l->forward(fbuf);
        }
        return TensorView{fbuf};
    }

    
    Tensor Sequential::backward(const Tensor& grad_out) {
        Tensor g = grad_out;
        for (std::size_t i = layers.size(); i-- > 0; ) {
            g = layers[i]->backward(g); 
        }
        return g;
    }

    TensorView Sequential::backward() {
        for (std::size_t i = layers.size(); i-- > 0; ) {
            bbuf = layers[i]->backward(bbuf); 
        }
        return TensorView{bbuf};
    }
    
    void Sequential::step(float lr, size_t batch_size) { // learn rate
        for (size_t i = layers.size() - 1; i < layers.size(); --i) {
            layers[i]->step(lr, batch_size);
        }
    }

    TensorView Sequential::grad_loss(const TensorView& a, const TensorView& b) { // Gradient of loss w.r.t output
        // Input tensor size: batch_size x feature_dim
        size_t a_size = a.rows * a.cols;
        std::vector<float> out(a_size);
        for (size_t i = 0; i < a_size; i++) {
            out[i] = a.data[i] - b.data[i];
        }
        bbuf = Tensor(out, a.rows, a.cols);
        return TensorView(bbuf);
    }

}