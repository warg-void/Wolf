// This file is separated from the rest to disable the compiler option ffast-math
// which will cause nans in the computation.
#include <model/LinearLayer.h>
#include <cmath>

namespace wolf{
    void LinearLayer::step_Adam(float lr, float beta1, float beta2, float eps, float bc1, float bc2, size_t batch_size) {
        const float inv_beta1 = 1.0f / bc1;
        const float inv_beta2 = 1.0f / bc2;
        const float scaled = inv_beta1 * lr / static_cast<float>(batch_size);


        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < W.size(); i_++) {
            size_t i = static_cast<size_t>(i_);
            vW(i) = beta1 * vW(i) + (1.0f - beta1) * dW(i);
            rW(i) = beta2 * rW(i) + (1.0f - beta2) * dW(i) * dW(i);
            const float r_hat = rW(i) * inv_beta2;
            const float denom = eps + std::sqrt(r_hat);
            W(i) -= scaled * vW(i) / denom;
            dW(i) = 0.0f;
        }
        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < b.size(); i_++) {
            size_t i = static_cast<size_t>(i_);
            vb(i) = beta1 * vb(i) + (1.0f - beta1) * db(i);
            rb(i) = beta2 * rb(i) + (1.0f - beta2) * db(i) * db(i);
            const float r_hat = rb(i) * inv_beta2;
            const float denom = eps + std::sqrt(r_hat);
            b(i) -= scaled * vb(i) / denom;
            db(i) = 0.0f;
        }
    }
}