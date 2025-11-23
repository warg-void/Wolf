#include <model/Sequential.h>
#include <external/zpp_bits.h>
#include <fstream>
#include <model/LayerSaver.h>
#include <model/optimizers.h>
#include <cmath>

namespace wolf {
    void Sequential::set_optimizer(OptimVariant cfg) {
        optim_cfg = std::move(cfg);
        step_t = 0;
    }

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

    
    Tensor Sequential::backward(const Tensor& grad_y) {
        Tensor g = grad_y;
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

    void Sequential::step(size_t batch_size) {
        if (!optim_cfg) {
            throw std::runtime_error("Optimizer not set");
        }
        std::visit([&](auto& opt){
            using Opt = std::decay_t<decltype(opt)>;
            if constexpr (std::is_same_v<Opt, SGD>) {
                for (auto& l : layers) {
                        l->step_SGD(opt.lr, batch_size);
                    }
            } else if constexpr (std::is_same_v<Opt, RMSProp>) {
                    for (auto& l : layers) {
                        l->step_RMSProp(opt.lr, opt.alpha, opt.eps, batch_size);
                    }
            } else if constexpr(std::is_same_v<Opt, Momentum>) {
                    for (auto& l : layers) {
                         l->step_momentum(opt.lr, opt.mu, batch_size);
                    }
            }
                
            else if constexpr (std::is_same_v<Opt, Adam>) {
                ++step_t;
                const float bc1 = 1.0f - std::pow(opt.beta1, static_cast<float>(step_t));
                const float bc2 = 1.0f - std::pow(opt.beta2, static_cast<float>(step_t));

                for (auto& l : layers) {
                    l->step_Adam(opt.lr, opt.beta1, opt.beta2, opt.eps,
                                bc1, bc2, batch_size);
                }
            }
        }, *optim_cfg);

    }

    TensorView Sequential::compute_grad_loss(const TensorView& a, const TensorView& b) { // Gradient of loss w.r.t output
        // Input tensor size: batch_size x feature_dim
        size_t a_rows = a.rows, a_cols = a.cols;
        size_t a_size = a.rows * a.cols;
        std::vector<float> out(a_size);
        switch (loss_cfg.l) {
            case LossType::MSE:
                for (size_t i = 0; i < a_size; i++) {
                    out[i] = a.data[i] - b.data[i];
                }
                break;
            case LossType::CrossEntropy: {
                for (size_t i = 0; i < a_rows; ++i) {
                    const size_t row = i * a_cols;

                    float m = a.data[row];
                    for (size_t j = 1; j < a_cols; ++j) {
                        m = std::max(m, a.data[row + j]);
                    }
                    float sum = 0.0f;
                    for (size_t j = 0; j < a_cols; ++j) {
                        float e = std::exp(a.data[row + j] - m);
                        out[row + j] = e; 
                        sum += e;
                    }
                    const float inv_sum = 1.0f / sum;
                    for (size_t j = 0; j < a_cols; ++j) {
                        float p = out[row + j] * inv_sum;
                        out[row + j] = p - b.data[row + j];
                    }
                }
                break;
            }
            case LossType::BCEWithLogits: {
                for (size_t i = 0; i < a_size; i++) {
                    // Sigmoid activation
                    float z = a.data[i];
                    float y;
                    if (z >= 0.0f) {
                        float ez = std::exp(-z);
                        y = 1.0f / (1.0f + ez);
                    } else {
                        float ez = std::exp(z);
                        y = ez / (1.0f + ez);
                    }
                    out[i] = y - b.data[i];
                }
            }
        }
        bbuf = Tensor(out, a.rows, a.cols);
        return TensorView(bbuf);
    }

    void Sequential::save(const std::string &path) const {
        auto [data, out] = zpp::bits::data_out();
        std::size_t n = layers.size();
        out(n).or_throw();
        
        for (const auto &ptr : layers) {
            save_layer(out, *ptr);
        }

        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Sequential::save: failed to open " + path);
        }

        file.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
        if (!file) {
            throw std::runtime_error("Sequential::save: failed to write " + path);
        }
    }


    Sequential Sequential::load(const std::string &path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Sequential::load: failed to open " + path);
        }

        std::streampos end = file.tellg();
        if (end < 0) {
            throw std::runtime_error("Sequential::load: tellg() failed for " + path);
        }

        size_t size = static_cast<size_t>(end);
        file.seekg(0, std::ios::beg);

        std::vector<std::byte> data(size);
        if (!file.read(reinterpret_cast<char*>(data.data()),
                    static_cast<std::streamsize>(size))) {
            throw std::runtime_error("Sequential::load: failed to read " + path);
        }

        zpp::bits::in in(data);

        size_t n{};
        in(n).or_throw();

        Sequential seq;
        seq.layers.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            seq.layers.emplace_back(load_layer(in));
        }

        return seq;
    }

}