#pragma once
#include <math/tensor.h>
#include <vector>
#include <memory>
#include <external/zpp_bits.h>

namespace wolf {

enum class LayerKind : uint8_t {
    Linear,
    ReLU,
};
class Layer {
public:
    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& grad_out) = 0; // input: gradient of the output, output: gradient of the input 
    
    virtual void step(float lr, size_t batch_size) = 0;
    virtual ~Layer() = default;
    LayerKind kind() const noexcept { return _kind; }
    virtual void save_body(zpp::bits::out<std::vector<std::byte>>& out) const = 0;
protected:
    explicit Layer(LayerKind k) : _kind(k) {}
private:
    LayerKind _kind;
};

}
