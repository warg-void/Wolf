#include <wolf.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <print>
#include <chrono>

using namespace wolf;

struct MnistSample {
    std::vector<float> x;  // 784 floats in [0,1]
    int label;             // 0..9
};

std::vector<MnistSample> load_mnist_csv(const std::string& path, int max_samples = -1) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open " + path);
    }

    std::vector<MnistSample> data;
    std::string line;

    // Skip header
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty MNIST file: " + path);
    }

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string field;

        MnistSample s;
        s.x.resize(784);

        // label
        if (!std::getline(ss, field, ',')) {
            continue; // malformed line
        }
        s.label = std::stoi(field);

        // 784 pixels
        for (int i = 0; i < 784; ++i) {
            if (!std::getline(ss, field, ',')) {
                throw std::runtime_error("Malformed MNIST row (not enough pixels) in " + path);
            }
            int pix = std::stoi(field);
            s.x[i] = static_cast<float>(pix) / 255.0f;  // normalize to [0,1]
        }

        data.push_back(std::move(s));

        if (max_samples > 0 && static_cast<int>(data.size()) >= max_samples) {
            break;
        }
    }

    return data;
}

// Convert sample to input Tensor [1 x 784]
Tensor make_input(const MnistSample& s) {
    return Tensor(s.x, 1, 784);
}

// One-hot target [1 x 10]
Tensor make_target(const MnistSample& s) {
    std::vector<float> t(10, 0.0f);
    t[s.label] = 1.0f;
    return Tensor(t, 1, 10);
}


int main(int argc, char** argv) {
    // retrieves dataset from <build>/examples/data/mnist_test.csv
    // dataset from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    std::filesystem::path exe_path = std::filesystem::canonical(argv[0]);
    std::filesystem::path exe_dir  = exe_path.parent_path();
    std::filesystem::path data_dir = exe_dir / "data";

    std::filesystem::path train_path = data_dir / "mnist_train.csv";
    std::filesystem::path test_path  = data_dir / "mnist_test.csv";

    std::println("Executable dir: {}", exe_dir.string());
    std::println("Loading MNIST train from: {}", train_path.string());
    std::println("Loading MNIST test  from: {}", test_path.string());

    auto train_data = load_mnist_csv(train_path.string());
    auto test_data  = load_mnist_csv(test_path.string());

    std::println("Loaded {} train samples, {} test samples",
                 train_data.size(), test_data.size());

    // Build 2 layer model: 784 -> 128 -> ReLU -> 10
    Sequential model(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    );

    float lr     = 0.01f;
    int   epochs = 3;          // Number of times the model is trained over whole train set (repeat)

    std::mt19937 gen(std::random_device{}());

    // Training
    std::vector<std::size_t> indices(train_data.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::shuffle(indices.begin(), indices.end(), gen);

        float epoch_loss = 0.0f;

        for (std::size_t k = 0; k < indices.size(); ++k) {

            const auto& s = train_data[indices[k]];

            Tensor x = make_input(s);
            Tensor t = make_target(s);
            

            Tensor y = model.pred(x);

            float loss = total_mse_loss(y, t);
            Tensor dE_dy = grad_loss(y, t);
            

            model.backward(dE_dy);
            model.step(lr);

            epoch_loss += loss;
            if ((k + 1) % 10000 == 0) {
                std::chrono::steady_clock::time_point cur_time = std::chrono::steady_clock::now();
                std::println("Epoch {} step {}/{} - running avg loss = {} t = {}",
                             epoch, k + 1, indices.size(),
                             epoch_loss / static_cast<float>(k + 1),
                             std::chrono::duration_cast<std::chrono::seconds> (cur_time - begin).count());
            }
        }

        float avg_loss = epoch_loss / static_cast<float>(indices.size());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::println("Epoch {} finished. Avg loss = {} runtime epoch = {}ms", epoch, avg_loss, std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count());
    }

    // Evaluation on test set
    int correct = 0;
    for (const auto& s : test_data) {
        Tensor x = make_input(s);
        Tensor y = model.pred(x);

        const auto& yr = y.raw();
        int pred = 0;
        for (int i = 1; i < 10; ++i) {
            if (yr[i] > yr[pred]) pred = i;
        }

        if (pred == s.label) ++correct;
    }

    float acc = 100.0f * static_cast<float>(correct) / static_cast<float>(test_data.size());
    std::println("Test accuracy: {}/{} ({:.2f}%)",
                 correct, test_data.size(), acc);

    return 0;
}
