#pragma once

#include <span>
#include <random>
#include <math/tensor.h>
#include <fstream>
#include <external/zpp_bits.h>
namespace wolf{

    TensorView make_batch_view(std::span<float> x, size_t x_dim, size_t sample_number, size_t batch_size) {
        size_t offset = sample_number * x_dim;
        return TensorView{x.data() + offset, batch_size, x_dim};
    }

    inline TensorView make_batch_view_indexed(
    std::span<float> data,                // flattened data
    std::size_t dim,                      // features per sample
    std::span<const std::size_t> indices, // permutation
    std::size_t start_sample,             // index in indices
    std::size_t batch_size,
    Tensor& batch_buf                     // output data location
    ) {
        const std::size_t num_samples = data.size() / dim;
        if (num_samples == 0 || batch_size == 0) {
            return TensorView{nullptr, 0, dim};
        }

        if (indices.size() < num_samples) {
            throw std::runtime_error("indices.size() < num_samples in make_batch_view_indexed");
        }

        const std::size_t needed = batch_size * dim;

        // Ensure buffer has enough storage
        if (batch_buf.empty() || batch_buf.data().size() < needed) {
            batch_buf = Tensor(std::move(std::vector<float>(needed)), batch_size, dim);
        }

        auto& buf = batch_buf.data();

        for (std::size_t i = 0; i < batch_size; ++i) {
            std::size_t sample_idx = indices[start_sample + i];
            const float* src = data.data() + sample_idx * dim;
            float* dst = buf.data() + i * dim;
            std::copy_n(src, dim, dst);
        }

        batch_buf.set_rows(batch_size);
        batch_buf.set_cols(dim);

        return TensorView{batch_buf};
    }

    struct BatchMaker {
        Tensor x_buf;
        Tensor t_buf;
        std::vector<std::size_t> indices;

        BatchMaker(size_t num_samples)
            : indices(num_samples) {
            std::iota(indices.begin(), indices.end(), 0);
        }

        void shuffle(std::uniform_random_bit_generator auto& gen) {
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        TensorView x_batch(std::span<float> x_data,
                        size_t x_dim,
                        size_t start_sample,
                        size_t batch_size) {
            return make_batch_view_indexed(
                x_data, x_dim,
                std::span<const size_t>(indices),
                start_sample, batch_size,
                x_buf
            );
        }

        TensorView t_batch(std::span<float> t_data,
                        size_t t_dim,
                        size_t start_sample,
                        size_t batch_size) {
            return make_batch_view_indexed(
                t_data, t_dim,
                std::span<const size_t>(indices),
                start_sample, batch_size,
                t_buf
            );
        }

        // Tensor make_batch(std::span<float> t_data,
        //                 size_t start_sample,
        //                 size_t batch_size) {
        //     indices[start_sample];
        //     return Tensor(std::vector<float>{}, 1, batch_size);
        // }
    };

    void save_tensor(const wolf::Tensor& t, const std::string& path) {
        auto [data, out] = zpp::bits::data_out();

        const std::size_t rows = t.nrows();
        const std::size_t cols = t.ncols();
        const std::vector<float> raw      = t.data();

        // Serialize (rows, cols, data) in that order
        out(rows, cols, raw).or_throw();

        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("failed to create std::ofstream for: " + path);
        }

        file.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
        if (!file) {
            throw std::runtime_error("failed to write tensor to file: " + path);
        }
    }

    Tensor load_tensor(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        size_t size = static_cast<std::size_t>(file.tellg());

        file.seekg(0, std::ios::beg);

        std::vector<std::byte> data(size);
        if (!file.read(reinterpret_cast<char*>(data.data()),
                    static_cast<std::streamsize>(size))) {
            throw std::runtime_error("Failed to read file: " + path);
        }

        zpp::bits::in in(data);

        size_t rows = 0;
        size_t cols = 0;
        std::vector<float> raw;

        in(rows, cols, raw).or_throw();

        return wolf::Tensor(std::move(raw), rows, cols);
    }

    void export_tensor_csv(const wolf::Tensor& t, const std::string& path) {
        const std::size_t rows = t.nrows();
        const std::size_t cols = t.ncols();
        const std::vector<float>& raw = t.data();

        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("failed to create std::ofstream for: " + path);
        }

        for (std::size_t r = 0; r < rows; ++r) {
            const std::size_t base = r * cols;
            for (std::size_t c = 0; c < cols; ++c) {
                file << raw[base + c];
                if (c + 1 < cols) {
                    file << ',';
                }
            }
            file << '\n';
        }

        if (!file) {
            throw std::runtime_error("failed to write tensor to CSV file: " + path);
        }
    }




}