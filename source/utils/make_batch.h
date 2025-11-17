#include <math/tensor.h>

namespace wolf {
    Tensor make_batch(std::vector<std::vector<float>> input) { // Ta
        std::vector<float> out;
        for (auto &v : input) {
            out.insert(out.end(), v.cbegin(), v.cend());
        }
        return Tensor(out, input.size(), input[0].size());
    }
    Tensor make_batch(std::vector<float> input, int batch_size) {
        return Tensor(input, batch_size, input.size() / batch_size);
    }
}