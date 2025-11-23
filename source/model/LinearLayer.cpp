#include <model/LinearLayer.h>
#include <math/rng.h>
#include <algorithm>
#include <utils/timer.h>
namespace wolf {
    LinearLayer::LinearLayer(size_t x_dim, size_t y_dim) : Layer(LayerKind::Linear), x_dim(x_dim),
            y_dim(y_dim) {
        auto& gen = rng().gen;
        auto normal_gen = [&]() {return std::normal_distribution<float>{0.0f, std::sqrt(2.0f / x_dim)}(gen);};
        std::vector<float> temp(y_dim * x_dim);
        std::vector<float> temp_b(y_dim);
        std::ranges::generate(temp, normal_gen);
        std::ranges::generate(temp_b, normal_gen);
        W = Tensor(temp, y_dim, x_dim);
        dW = Tensor(std::vector<float>(y_dim * x_dim, 0.0f), y_dim, x_dim);
        vW = Tensor(std::vector<float>(y_dim * x_dim, 0.0f), y_dim, x_dim);
        b = Tensor(temp_b, 1, y_dim);
        vb = Tensor(std::vector<float>(y_dim, 0.0f), 1, y_dim);
        db = Tensor(std::vector<float>(y_dim, 0.0f), 1, y_dim);
        rW = Tensor(std::vector<float>(y_dim * x_dim, 0.0f), y_dim, x_dim);
        rb = Tensor(std::vector<float>(y_dim, 0.0f), 1, y_dim);
    }
    Tensor LinearLayer::forward(const Tensor& x) {
        last_input = x;
        size_t batch_size = x.nrows();
        std::vector<float> out(batch_size * y_dim);
        #pragma omp parallel for 
        for (std::ptrdiff_t j_ = 0; j_ < y_dim * batch_size; j_++) {
            size_t j = static_cast<size_t>(j_);
            const size_t k  = j % y_dim;      // output neuron index
            const size_t bn = j / y_dim;      // batch index
            float sum = b(k);
            for (size_t i = 0; i < x_dim; ++i) {
                sum += x(i + x_dim * bn) * W(k, i);
            }
            out[j] = sum;
        }
        return Tensor(out, batch_size, y_dim);
    }

    Tensor LinearLayer::backward(const Tensor& grad_out) {
        size_t batch_size = grad_out.nrows();
        std::vector<float> grad_in(x_dim * batch_size, 0.0f);

        // Unparallelized version of the below code:
        // for (size_t i = 0; i < y_dim; i++) {
        //     db(i) = grad_in(i) / (float)batch_size;
        //     for (size_t j = 0; j < x_dim; j++) {
        //         dW[i * x_dim + j] = grad_in(i) * x[j] / (float)batch_size;
        //         grad_in[j] += W[i * x_dim + j] * grad_out(i); 
        //     }
        // }

        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < y_dim; i_++) {
            size_t y = static_cast<size_t>(i_);
            float db_acc = 0.0f;
            const size_t y_flatten = y * x_dim;

            for (size_t sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
                const float this_sample_grad_out = grad_out(sample_idx * y_dim + y);
                db_acc += this_sample_grad_out;
                const size_t sample_in_offset = sample_idx * x_dim;
                for (size_t x = 0; x < x_dim; ++x) {
                        dW(y_flatten + x) += this_sample_grad_out * last_input(x + sample_in_offset);
                }
            }

            db(y) = db_acc;
        }
        #pragma omp parallel for 
        for (std::ptrdiff_t j_ = 0; j_ < x_dim; j_++) {
            size_t x = static_cast<size_t>(j_);
            for (size_t sample_idx = 0; sample_idx < batch_size; ++sample_idx) {    
                float sum = 0.0f;
                const size_t sample_out_offset = sample_idx * y_dim;
                for (size_t y = 0; y < y_dim; ++y) {
                    sum += W(y * x_dim + x) * grad_out(y + sample_out_offset);
                }
                grad_in[sample_idx * x_dim + x] = sum;  
            }
        };
        return Tensor(grad_in, batch_size, x_dim);

    }

    void LinearLayer::step_SGD(float lr, size_t batch_size) {
        const float scale = lr / static_cast<float>(batch_size);
        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < W.size(); i_++) {
            size_t W_idx = static_cast<size_t>(i_);
            W(W_idx) -= scale * dW(W_idx) ;
            dW(W_idx) = 0.0f;
        }

        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < b.size(); i_++) {
            size_t b_idx = static_cast<size_t>(i_);
            b(b_idx) -= scale * db(b_idx);
            db(b_idx) = 0.0f;
        }

    }

    void LinearLayer::step_momentum(float lr, float mu, size_t batch_size) {
        const float scale = lr / static_cast<float>(batch_size);
        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < W.size(); i_++) {
            size_t W_idx = static_cast<size_t>(i_);
            vW(W_idx) = -scale * dW(W_idx) + mu * vW(W_idx);
            W(W_idx) += vW(W_idx);
            dW(W_idx) = 0.0f;
        }

        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < b.size(); i_++) {
            size_t b_idx = static_cast<size_t>(i_);
            vb(b_idx) = -scale * db(b_idx) + mu * vb(b_idx);
            b(b_idx) += vb(b_idx);
            db(b_idx) = 0.0f;
        }
    }

    void LinearLayer::step_RMSProp(float lr, float alpha, float eps, size_t batch_size) {
        const float scale = lr / static_cast<float>(batch_size);
        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < W.size(); i_++) {
            size_t W_idx = static_cast<size_t>(i_);
            rW(W_idx) = alpha * rW(W_idx) +  (1.0f - alpha) * dW(W_idx) * dW(W_idx);
            W(W_idx) -= scale * dW(W_idx) / (std::sqrt(rW(W_idx)) + eps);
            dW(W_idx) = 0.0f;
        }

        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < b.size(); i_++) {
            size_t b_idx = static_cast<size_t>(i_);
            rb(b_idx) = alpha * rb(b_idx) + (1.0f - alpha) * db(b_idx) * db(b_idx);
            b(b_idx) -= scale * db(b_idx) / (std::sqrt(rb(b_idx)) + eps);
            db(b_idx) = 0.0f;
        }
    }

    // Step Adam in AdamStepper.cpp due to floating math restrictions
}