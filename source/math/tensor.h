#pragma once
#include <vector>
#include <print>
namespace wolf {



class Tensor {
private:
    std::vector<float> data_;
    size_t rows{0};
    size_t cols{0};
public:
    Tensor() = default;
    Tensor(const std::vector<float>& v, size_t r, size_t c)
        : data_(v), rows(r), cols(c) {}
    Tensor(std::vector<float>&& v, size_t r, size_t c)
        : data_(std::move(v)), rows(r), cols(c) {}    
    Tensor(const std::vector<std::vector<float>>& input);
    size_t nrows() const {return rows;}
    size_t ncols() const {return cols;}
    void set_rows(size_t r) {rows = r;}
    void set_cols(size_t c) {cols = c;}

    std::vector<float>& raw() {return data_;}
    const std::vector<float>& raw() const { return data_; }

    template <class Self>
    auto&& data(this Self&& self) {
        return std::forward<Self>(self).data_;
    }
    inline float operator()(size_t r, size_t c) const {
        return data_[static_cast<std::size_t>(r) * cols + c];
    };
    inline float& operator()(size_t r, size_t c) {
        return data_[r * cols + c];
    }
    inline float operator()(size_t i) const {
        return data_[i];
    }
    inline float& operator()(size_t i) {
        return data_[i];
    };
    inline float operator[](size_t i) const {
        return data_[i];
    }
    inline float& operator[](size_t i) {
        return data_[i];
    };
    bool empty() const {
        return rows == 0 && cols == 0;
    }
    size_t size() const {
        return rows * cols;
    }
};
 

struct TensorView {
    float* data;
    size_t rows, cols;

    TensorView(float* data, size_t rows, size_t cols) : data(data), rows(rows), cols(cols) {}
    TensorView(Tensor& input) : data(input.raw().data()), rows(input.nrows()), cols(input.ncols()) {}

    float& operator()(std::size_t i)       { return data[i]; }
    float  operator()(std::size_t i) const { return data[i]; }

    float& operator()(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }
    float operator()(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }
};
}
