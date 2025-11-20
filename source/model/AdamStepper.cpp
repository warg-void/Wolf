// This file is separated from the rest to disable the compiler option ffast-math
// which will cause nans in the computation.
#include <model/LinearLayer.h>
#include <cmath>

namespace wolf{
    void LinearLayer::step_Adam(float lr, float beta1, float beta2, float eps, float bc1, float bc2, size_t batch_size) {
        auto& W_raw = W.raw();
        auto& dW_raw = dW.raw();
        auto& rW_raw = rW.raw();
        auto& b_raw = b.raw();
        auto& db_raw = db.raw();
        auto& rB_raw = rB.raw();
        auto& vW_raw = vW.raw(); // v is used as s here
        auto& vB_raw = vB.raw();
        const float inv_bc1 = 1.0f / bc1;
        const float inv_bc2 = 1.0f / bc2;
        const float scaled = inv_bc1 * lr / static_cast<float>(batch_size);


        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < W_raw.size(); i_++) {
            size_t i = static_cast<size_t>(i_);
            vW_raw[i] = beta1 * vW_raw[i] + (1.0f - beta1) * dW_raw[i];
            rW_raw[i] = beta2 * rW_raw[i] + (1.0f - beta2) * dW_raw[i] * dW_raw[i];
            const float r_hat = rW_raw[i] * inv_bc2;
            const float denom = eps + std::sqrt(r_hat);
            W_raw[i] -= scaled * vW_raw[i] / denom;
            dW_raw[i] = 0.0f;
        }
        #pragma omp parallel for 
        for (std::ptrdiff_t i_ = 0; i_ < b_raw.size(); i_++) {
            size_t i = static_cast<size_t>(i_);
            vB_raw[i] = beta1 * vB_raw[i] + (1.0f - beta1) * db_raw[i];
            rB_raw[i] = beta2 * rB_raw[i] + (1.0f - beta2) * db_raw[i] * db_raw[i];
            const float r_hat = rB_raw[i] * inv_bc2;
            const float denom = eps + std::sqrt(r_hat);
            b_raw[i] -= scaled * vB_raw[i] / denom;
            db_raw[i] = 0.0f;
        }
    }
}