#pragma once

#include <memory>
#include <stdexcept>
#include <external/zpp_bits.h>

#include <model/Layer.h>
#include <model/LinearLayer.h>
#include <model/ReLU.h>
#include <external/zpp_bits.h>

namespace wolf {

    inline void save_layer(zpp::bits::out<std::vector<std::byte>>& out,
                    const Layer& layer) {
        LayerKind kind = layer.kind();
        out(kind).or_throw();
        layer.save_body(out);
    }

    // Read one Layer from a zpp_bits archive.
    inline std::unique_ptr<Layer> load_layer(zpp::bits::in<std::vector<std::byte>>& in) {
        LayerKind kind{};
        in(kind).or_throw();

        switch (kind) {
        case LayerKind::Linear:
            return LinearLayer::load_from(in);
        case LayerKind::ReLU:
            return ReLULayer::load_from(in);
        default:
            throw std::runtime_error("load_layer: unknown LayerKind");
        }
    }
}