#include <wolf.h>
#include <print>
#include <random>

using namespace wolf;

int main() {
    struct Sample { float x1, x2, y; };
    // Nonlinear XOR data
    std::vector<Sample> data = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f},
    };

    // Model: 2 -> 16 -> 1
    Sequential model(
        Linear(2, 16),
        ReLU(),
        Linear(16, 1)
    );

    float lr = 0.1f;
    int epochs = 50000;

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(0, (int)data.size() - 1);

    for (int step = 0; step < epochs; ++step) {
        // pick random sample
        int idx = dist(gen);
        auto s = data[idx];

        Tensor x({s.x1, s.x2}, 1, 2); 
        Tensor t({s.y}, 1, 1);         

        Tensor y = model.pred(x);       // forward
        Tensor dE_dy = grad_loss(y, t);    // mse loss gradient w.r.t. y

        model.backward(dE_dy);          // backprop through layers
        model.step(lr);                 // SGD update

        if (step % 5000 == 0) {
            std::println("step {}: sample {}, x=({}, {}), y={}, t={}, dEdy={}",
                         step, idx, s.x1, s.x2, y(0,0), t(0,0), dE_dy(0));
        }
    }

    // Test all 4 after training
    std::println("\nFinal predictions:");
    for (auto s : data) {
        Tensor x({s.x1, s.x2}, 1, 2);
        Tensor t({s.y}, 1, 1);
        Tensor y = model.pred(x);
        std::println("x=({}, {}), y_pred={:.4f}, target={}",
                     s.x1, s.x2, y(0,0), t(0,0));
    }
}
