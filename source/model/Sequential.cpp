#include <model/Sequential.h>

namespace wolf {
    Tensor Sequential::pred(const Tensor& x) {
        Tensor out = x;
        for (auto& l : layers) {
            out = l->forward(out);
        }
        return out;
    }
}