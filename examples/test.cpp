#include <wolf.h>

using namespace wolf;

int main() {

    std::vector<float> x_data;
    std::vector<float> t_data;
    std::mt19937 gen(std::random_device{}());
    size_t x_dim = 10;
    size_t t_dim = 4;
    float lr = 5;
    size_t batch_size = 10;
    size_t epochs = 5;
    size_t num_samples = x_data.size() / x_dim;

    auto model = Sequential(Linear(3, 5));

    for (size_t n = 0; n < epochs; n++) {
        shuffle_dataset(x_data, t_data, x_dim, t_dim, gen);

        for (size_t s = 0; s < num_samples; s += batch_size) {
            size_t current_bs = std::min(batch_size, num_samples - s);
            TensorView x_batch = make_batch_view(x_data, x_dim, s, current_bs);
            TensorView t_batch = make_batch_view(t_data, t_dim, s, current_bs);
            TensorView y_batch = model.pred(x_batch);
            model.grad_loss(y_batch, t_batch);
            model.backward();
            model.step(lr, current_bs);

            // For printing
            float loss = total_mse_loss(y_batch, t_batch);
        }
    }
}