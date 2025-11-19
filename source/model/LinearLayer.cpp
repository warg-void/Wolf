#include <model/LinearLayer.h>
#include <math/rng.h>
#include <algorithm>
#include <execution>
#include <utils/timer.h>
namespace wolf {
    LinearLayer::LinearLayer(size_t in_dim, size_t out_dim) : Layer(LayerKind::Linear), in_dim(in_dim),
            out_dim(out_dim) {
        auto& gen = rng().gen;
        auto normal_gen = [&]() {return std::normal_distribution{0.0, std::sqrt(2.0 / in_dim)}(gen);};
        std::vector<float> temp(out_dim * in_dim);
        std::vector<float> temp_b(out_dim);
        std::ranges::generate(temp, normal_gen);
        std::ranges::generate(temp_b, normal_gen);
        W = Tensor(temp, out_dim, in_dim);
        dW = Tensor(std::vector<float>(out_dim * in_dim, 0.0f), out_dim, in_dim);
        b = Tensor(temp_b, 1, out_dim);
        db = Tensor(std::vector<float>(out_dim, 0.0f), 1, out_dim);
    }
    Tensor LinearLayer::forward(const Tensor& x) {
        last_input = x;
        int batch_size = x.nrows();
        std::vector<float> out(batch_size * out_dim);
        #pragma omp parallel for 
        for (size_t j = 0; j < out_dim * batch_size; j++) {
            const size_t k  = j % out_dim;      // output neuron index
            const size_t bn = j / out_dim;      // batch index
            float sum = b(k);
            for (size_t i = 0; i < in_dim; ++i) {
                sum += x(i + in_dim * bn) * W(k, i);
            }
            out[j] = sum;
        }
        return Tensor(out, batch_size, out_dim);
    }

    Tensor LinearLayer::backward(const Tensor& grad_out) {
        size_t batch_size = grad_out.nrows();
        const auto& x = last_input.raw();
        const auto& gy = grad_out.raw();
        auto& dW_raw = dW.raw();
        auto& db_raw = db.raw();
        const auto& W_raw = W.raw();
        std::vector<float> gx(in_dim * batch_size, 0.0f);
        // Unparallelized version of the below code:
        // for (size_t i = 0; i < out_dim; i++) {
        //     db_raw[i] = gy[i] / (float)batch_size;
        //     for (size_t j = 0; j < in_dim; j++) {
        //         dW_raw[i * in_dim + j] = gy[i] * x[j] / (float)batch_size;
        //         gx[j] += W_raw[i * in_dim + j] * gy[i]; 
        //     }
        // }
        #pragma omp parallel for 
        for (size_t i = 0; i < out_dim; i++) {
            float db_acc = 0.0f;
            const size_t row = i * in_dim;

            for (size_t b = 0; b < batch_size; ++b) {
                const float gy_bi = gy[b * out_dim + i];
                db_acc += gy_bi;

                const float* x_b = &x[b * in_dim];
                for (size_t j = 0; j < in_dim; ++j) {
                        dW_raw[row + j] += gy_bi * x_b[j];
                }
            }

            db_raw[i] = db_acc;
        }
        #pragma omp parallel for 
        for (size_t j = 0; j < in_dim; j++) {
            for (size_t b = 0; b < batch_size; ++b) {
                const float* gy_b = &gy[b * out_dim];        
                float sum = 0.0f;
                for (size_t i = 0; i < out_dim; ++i) {
                    sum += W_raw[i * in_dim + j] * gy_b[i];
                }
                gx[b * in_dim + j] = sum;  
            }
        };


        return Tensor(gx, batch_size, in_dim);

    }

    void LinearLayer::step(float lr, size_t batch_size) {
        // SGD
        auto& W_raw = W.raw();
        auto& dW_raw = dW.raw();
        auto& b_raw = b.raw();
        auto& db_raw = db.raw();
        const float scale = lr / static_cast<float>(batch_size);
        #pragma omp parallel for 
        for (size_t i = 0; i < W_raw.size(); i++) {
            W_raw[i] -= scale * dW_raw[i];
            dW_raw[i] = 0.0f;
        }

        #pragma omp parallel for 
        for (size_t i = 0; i < b_raw.size(); i++) {
            b_raw[i] -= scale * db_raw[i];
            db_raw[i] = 0.0f;
        }

    }
}