#include <wolf.h> 
#include <print>
#include <random>
using namespace wolf;

int main() {
    int steps = 1000;
    float lr = 0.02; // Model will diverge at learning rates above 0.05
    std::vector<float> xs, ys;
    for (int i = -10; i <= 10; ++i) {
        float x = i;
        float y = 2.0f * x + 1.0f;
        xs.push_back(x);
        ys.push_back(y);
    }
    Sequential model(
        Linear(1, 1) // One neuron, one layer "neural net"
    );
    auto gen = std::mt19937(std::random_device{}());
    std::uniform_int_distribution<> dist(0, 19);
    for (int j = 0; j < steps; j++) {
        // for 1000 times, pick a random (x, y) and feed that into the model
        int i = dist(gen); 
        Tensor y = model.pred(Tensor({xs[i]}, 1, 1));
        Tensor t = Tensor({ys[i]}, 1, 1);
        Tensor gl = grad_loss(y, t); // gradient of the loss
        model.backward(gl);
        model.step(lr);
        std::println("x = {} y = {}, t = {}, gl = {} ", xs[i], y(0), t(0), gl(0));
    }
}