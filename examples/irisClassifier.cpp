#include <wolf.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <stdexcept>
#include <print> 
#include <filesystem> 

using namespace wolf;

struct IrisSample {
    float x[4];   // sepal_length, sepal_width, petal_length, petal_width
    int label;    // 0 = setosa, 1 = versicolor, 2 = virginica
};

std::vector<IrisSample> load_iris(const std::string& path) {
    std::vector<IrisSample> data;
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open " + path);
    }

    std::string line;

    // Skip header
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty iris file: " + path);
    }

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string field;

        IrisSample s;
        std::getline(ss, field, ','); s.x[0] = std::stof(field);
        std::getline(ss, field, ','); s.x[1] = std::stof(field);
        std::getline(ss, field, ','); s.x[2] = std::stof(field);
        std::getline(ss, field, ','); s.x[3] = std::stof(field);
        std::getline(ss, field, ',');
        if (field == "Iris-setosa")          s.label = 0;
        else if (field == "Iris-versicolor") s.label = 1;
        else                            s.label = 2;
        data.push_back(s);
    }

    return data;
}

// ----- Helpers to create tensors -----

Tensor make_input(const IrisSample& s) {
    // [1 x 4] row vector
    return Tensor({s.x[0], s.x[1], s.x[2], s.x[3]}, 1, 4);
}

Tensor make_target(const IrisSample& s) {
    // one-hot [1 x 3]
    std::vector<float> t(3, 0.0f);
    t[s.label] = 1.0f;
    return Tensor(t, 1, 3);
}

int main(int argc, char** argv) {
    // put iris.csv at data/iris.csv
    std::filesystem::path exe_path = std::filesystem::canonical(argv[0]);
    std::filesystem::path exe_dir  = exe_path.parent_path();

    std::filesystem::path iris_path = exe_dir / "data" / "iris.csv";

    std::println("Executable dir: {}", exe_dir.string());
    std::println("Loading iris from: {}", iris_path.string());

    auto data = load_iris(iris_path.string());
    std::println("Loaded {} iris samples", data.size());


    // 4 -> 8 -> ReLU -> 3
    Sequential model(
        Linear(4, 8),
        ReLU(),
        Linear(8, 3)
    );

    float lr    = 0.01f;
    int   steps = 5000;

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(0, static_cast<int>(data.size()) - 1);

    // Training loop (stochastic, 1 sample per step)
    for (int step = 0; step < steps; ++step) {
        int idx = dist(gen);
        const auto& s = data[idx];

        Tensor x = make_input(s);
        Tensor t = make_target(s);
        Tensor y = model.pred(x); 
        float loss = mse_loss(y, t);
        Tensor dE_dy = grad_loss(y, t);

        model.backward(dE_dy);
        model.step(lr);

        if (step % 500 == 0) {
            std::println("step {}: loss = {}", step, loss);
        }
    }

    // 4. Evaluate accuracy on the whole dataset
    int correct = 0;
    for (const auto& s : data) {
        Tensor x = make_input(s);
        Tensor y = model.pred(x);

        const auto& yr = y.raw();
        int pred = 0;
        for (int i = 1; i < 3; ++i) {
            if (yr[i] > yr[pred]) pred = i;
        }

        if (pred == s.label) ++correct;
    }

    float acc = 100.0f * static_cast<float>(correct) / static_cast<float>(data.size());
    std::println("Final accuracy: {}/{} ({:.2f}%)", correct, data.size(), acc);

    return 0;
}
