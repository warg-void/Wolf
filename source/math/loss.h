#include <math/tensor.h>

namespace wolf{
    Tensor loss(const Tensor& a, const Tensor& b) {
        const auto& a_raw = a.raw();
        const auto& b_raw = b.raw();
        std::vector<float> out(a_raw.size());
        for (size_t i = 0; i < a_raw.size(); i++) {
            out[i] = a_raw[i] - b_raw[i];
        }
        return Tensor(out, a.nrows(), a.ncols());
    }
}