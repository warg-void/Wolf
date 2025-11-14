#pragma once
#include <memory>
#include <model/Layer.h>
#include <model/LinearLayer.h>
#include <model/ReLU.h>

namespace wolf {
    inline std::unique_ptr<Layer> Linear(size_t in_dim, size_t out_dim) {
        return std::make_unique<LinearLayer>(in_dim, out_dim);
    }
    inline std::unique_ptr<Layer> ReLU() {
        return std::make_unique<ReLULayer>();
    }
}