#pragma once

#include <span>
#include <random>
#include <math/tensor.h>

namespace wolf{

    void shuffle_dataset(std::span<float> x_data, std::span<float> t_data, size_t x_dim, size_t t_dim,
        std::uniform_random_bit_generator auto& gen) {

        const std::size_t num_samples = x_data.size() / x_dim;
        if (num_samples == 0) return;

        const std::size_t num_samples_t = t_data.size() / t_dim;
        if (num_samples_t != num_samples) {
            throw std::runtime_error("x_data / x_dim is not equal to t_data / t_dim, the number of samples");
        }

        std::vector<size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        std::vector<float> x_tmp(x_data.size());
        std::vector<float> t_tmp(t_data.size());

        for (std::size_t new_i = 0; new_i < num_samples; ++new_i) {
            std::size_t old_i = indices[new_i];
            std::copy_n(x_data.begin() + old_i * x_dim,
                        x_dim,
                        x_tmp.begin() + new_i * x_dim);

            std::copy_n(t_data.begin() + old_i * t_dim,
                        t_dim,
                        t_tmp.begin() + new_i * t_dim);
        }

        std::copy(x_tmp.begin(), x_tmp.end(), x_data.begin());
        std::copy(t_tmp.begin(), t_tmp.end(), t_data.begin());
    }

    TensorView make_batch_view(std::span<float> x, size_t x_dim, size_t sample_number, size_t batch_size) {
        size_t offset = sample_number * x_dim;
        return TensorView{x.data() + offset, batch_size, x_dim};
    }
}