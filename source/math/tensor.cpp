#include <math/tensor.h>

namespace wolf {
    Tensor::Tensor(const std::vector<std::vector<float>>& input) {
        rows = input.size();
        if (rows == 0) {
            cols = 0;
            return;
        }
        cols = input[0].size();
        for (auto &i : input) {
            data.insert(data.end(), i.begin(), i.end());
        }
    }
}