#include <model/LinearLayer.h>
#include <math/rng.h>
#include <algorithm>
#include <execution>
#include <utils/timer.h>
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
        idx.resize(in_dim * out_dim);
        std::iota(idx.begin(), idx.end(), size_t{0});
    }
    Tensor LinearLayer::forward(const Tensor& x) {
        last_input = x;
        int xr = x.nrows(); // Number of batches
        std::vector<float> out(xr * out_dim);
        std::for_each(std::execution::par_unseq, idx.begin(), idx.begin() + out.size(),
            [&](size_t j){
                size_t k = j % out_dim; // Output Node index
                float sum = b(k);
                for (size_t i = 0; i < in_dim; i++) {
                    sum += x(i) * W(k, i);
                }
                out[j] = sum;
            }
        ); 
        return Tensor(out, xr, out_dim);
    }

    Tensor LinearLayer::backward(const Tensor& grad_out, int batch_size) {
        const auto& x = last_input.raw();
        const auto& gy = grad_out.raw();
        auto& dW_raw = dW.raw();
        auto& db_raw = db.raw();
        const auto& W_raw = W.raw();
        std::vector<float> gx(in_dim, 0.0f);
        // Unparallelized version of the below code:
        // for (size_t i = 0; i < out_dim; i++) {
        //     db_raw[i] = gy[i] / (float)batch_size;
        //     for (size_t j = 0; j < in_dim; j++) {
        //         dW_raw[i * in_dim + j] = gy[i] * x[j] / (float)batch_size;
        //         gx[j] += W_raw[i * in_dim + j] * gy[i]; 
        //     }
        // }
        const float inv_bs = 1.0f / static_cast<float>(batch_size);
        std::for_each(std::execution::par_unseq,
                    idx.begin(), idx.begin() + out_dim,
                    [&](size_t i) {
                        const float scale = gy[i] * inv_bs;  

                        db_raw[i] = scale; // = gy[i] / batch_size

                        const size_t row = i * in_dim;
                        for (size_t j = 0; j < in_dim; ++j) {
                            dW_raw[row + j] = scale * x[j];
                        }
                    });
        
        std::for_each(std::execution::par_unseq,
                    idx.begin(), idx.begin() + in_dim,
                    [&](size_t j) {
                        float sum = 0.0f;
                        for (size_t i = 0; i < out_dim; ++i) {
                            sum += W_raw[i * in_dim + j] * gy[i];
                        }
                        gx[j] += sum;
                    });


        return Tensor(gx, 1, in_dim);

    }

    void LinearLayer::step(float lr) {
        // SGD
        auto& W_raw = W.raw();
        auto& dW_raw = dW.raw();
        auto& b_raw = b.raw();
        auto& db_raw = db.raw();
        std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
            [&](size_t i) {
                W_raw[i] -= lr * dW_raw[i];
            }
        );
        std::for_each(std::execution::par_unseq, idx.begin(), idx.begin() + b_raw.size(),
            [&](size_t i) {
                b_raw[i] -= lr * db_raw[i];
            }
        );
    }
}