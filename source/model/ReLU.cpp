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
            return Tensor(out, x.nrows(), x.ncols());
        }

        Tensor ReLULayer::backward(const Tensor& grad_out) {
            std::vector<float> gx(last_input.size());
            for (size_t i = 0; i < last_input.size(); i++) {
                gx[i] = (last_input[i] > 0.0f) ? grad_out[i] : 0.0f;
            }
            return Tensor(gx, last_input.nrows(), last_input.ncols());
        }
    }