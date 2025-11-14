#include <model/LinearLayer.h>
#include <math/rng.h>
#include <algorithm>
namespace wolf {
    LinearLayer::LinearLayer(size_t in_dim, size_t out_dim) {
        this->in_dim = in_dim;
        this->out_dim = out_dim;

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
        std::vector<float> out(out_dim);
        for (size_t j = 0; j < out_dim; j++) {
            float sum = b(j);
            for (size_t i = 0; i < in_dim; i++) {
                sum += x(i) * W(j, i);
            }
            out[j] = sum;
        }
        return Tensor(out, 1, out_dim);
    }

    Tensor LinearLayer::backward(const Tensor& grad_out) {
        const auto& x = last_input.raw();
        const auto& gy = grad_out.raw();
        auto& dW_raw = dW.raw();
        auto& db_raw = db.raw();
        const auto& W_raw = W.raw();
        std::vector<float> gx(in_dim, 0.0f);
        for (size_t i = 0; i < out_dim; i++) {
            db_raw[i] = gy[i];
            for (size_t j = 0; j < in_dim; j++) {
                dW_raw[i * in_dim + j] = gy[i] * x[j]; 
            }
        }
        for (size_t i = 0; i < in_dim; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < out_dim; j++) {
                sum += W_raw[i + in_dim * j] * gy[j]; 
            }
            gx[i] = sum;
        }
        return Tensor(gx, 1, in_dim);

    }

    void LinearLayer::step(float lr) {
        // SGD
        auto& W_raw = W.raw();
        auto& dW_raw = dW.raw();
        auto& b_raw = b.raw();
        auto& db_raw = db.raw();
        // std::println("dW = {}", dW_raw[0]);
        for (size_t i = 0; i < W_raw.size(); i++) {
            W_raw[i] -= lr * dW_raw[i];
        }
        for (size_t i = 0; i < b_raw.size(); i++) {
            b_raw[i] -= lr * db_raw[i];
        }
        // std::println("W = {}, b = {}", W(0), b(0));
    }
}