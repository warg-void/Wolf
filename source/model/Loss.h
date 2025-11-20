#pragma once
#include <cstdint>
#include <cmath>

namespace wolf {

    enum class LossType : std::uint8_t {
        MSE,
        CrossEntropy,
        BCEWithLogits,
    };

    struct LossConfig {
        LossType l = LossType::MSE;
    };

    inline float mse_loss(const TensorView& a, const TensorView& b) {
        const auto& a_raw = a.data;
        const auto& b_raw = b.data;
        float out = 0.0f;
        for (size_t i = 0; i < a.cols * a.rows; i++) {
            out += 0.5 * (a_raw[i] - b_raw[i]) * (a_raw[i] - b_raw[i]);
        }
        return out;
    }

    inline float cross_entropy_loss(const TensorView& a, const TensorView& b) {
        const size_t a_rows = a.rows, a_cols = a.cols;
        const auto* a_ = a.data;
        const auto* b_ = b.data;

        float out = 0.0f;
        for (size_t i = 0; i < a_rows; ++i) {
            const size_t row = i * a_cols;

            float m = a_[row];
            for (size_t j = 1; j < a_cols; ++j)
                m = std::max(m, a_[row + j]);

            float sumexp = 0.0f;
            for (size_t j = 0; j < a_cols; ++j)
                sumexp += std::exp(a_[row + j] - m);

            const float logsumexp = m + std::log(sumexp + 1e-30f);

            float row_loss = 0.0f;
            for (size_t j = 0; j < a_cols; ++j) {
                const float bj = b_[row + j];
                if (bj != 0.0f) {
                    row_loss += bj * (logsumexp - a_[row + j]);
                }
            }
            out += row_loss;
        }
        return out;
    }

    inline float bce_with_logits_loss(const TensorView& a, const TensorView& b) {
        const auto* a_ = a.data;
        const auto* b_ = b.data;
        const size_t n = a.rows * a.cols;

        float out = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float ai = a_[i];
            const float bi = b_[i];
            const float absz = std::abs(ai);
            out += std::max(ai, 0.0f) - ai * bi + std::log1p(std::exp(-absz));
        }
        return out;
    }
    
}