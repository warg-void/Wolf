#pragma once

#include <math/tensor.h>

namespace wolf{
    Tensor grad_loss(const Tensor& a, const Tensor& b) { // Gradient of loss w.r.t output
        // Input tensor size: batch_size x feature_dim
        const auto& a_raw = a.raw();
        const auto& b_raw = b.raw();
        std::vector<float> out(a_raw.size());
        for (size_t i = 0; i < a_raw.size(); i++) {
            out[i] = a_raw[i] - b_raw[i];
        }
        return Tensor(out, a.nrows(), a.ncols());
    }

    float total_mse_loss(const Tensor& a, const Tensor& b) {
        const auto& a_raw = a.raw();
        const auto& b_raw = b.raw();
        float out = 0.0f;
        for (size_t i = 0; i < a_raw.size(); i++) {
            out += 0.5 * (a_raw[i] - b_raw[i]) * (a_raw[i] - b_raw[i]);
        }
        return out / a.nrows();
    }

    float total_mse_loss(const TensorView& a, const TensorView& b) {
        const auto& a_raw = a.data;
        const auto& b_raw = b.data;
        float out = 0.0f;
        for (size_t i = 0; i < a.cols * a.rows; i++) {
            out += 0.5 * (a_raw[i] - b_raw[i]) * (a_raw[i] - b_raw[i]);
        }
        return out / a.rows;
    }
}