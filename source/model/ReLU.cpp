    #include <model/ReLU.h>

    namespace wolf {
        Tensor ReLULayer::forward(const Tensor& x) {
        last_input = x;
        std::vector<float> out(x.ncols() * x.nrows());
        for (size_t i = 0; i < x.ncols() * x.nrows(); i++) {
            if (x(i) < 0) {
                out[i] = 0;
            } else {
                out[i] = x(i);
            }
        }
        return Tensor(std::move(out), x.nrows(), x.ncols());
        }

        Tensor ReLULayer::backward(const Tensor& grad_out) {
            const auto &x = last_input.raw();
            const auto &gy = grad_out.raw();

            std::vector<float> gx(x.size());
            for (size_t i = 0; i < x.size(); i++) {
                gx[i] = (x[i] > 0.0f) ? gy[i] : 0.0f;
            }
            return Tensor(std::move(gx), last_input.nrows(), last_input.ncols());
        }
    }