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

const int num_pixels = 784;
const int num_classes = 10;

std::size_t load_mnist_csv(
    const std::string& path,
    std::vector<float>& x_data,   // will contain N * 784 floats
    std::vector<float>& t_data   // will contain N labels as floats
) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open " + path);
    }
    std::string line;

    // Skip header
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty MNIST file");
    }


    std::size_t num_samples = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string field;
        if (!std::getline(ss, field, ',')) {
            continue;
        }

        int label = std::stoi(field);
        for (int i = 0; i < num_classes; i++) {
            t_data.push_back(i == label ? 1 : 0);
        }

        for (int i = 0; i < num_pixels; ++i) {
            if (!std::getline(ss, field, ',')) {
                throw std::runtime_error(
                    "Malformed MNIST row (not enough pixels) in " + path
                );
            }
            int pix = std::stoi(field);
            float norm_pix = static_cast<float>(pix) / 255.0f; // [0, 1]
            x_data.push_back(norm_pix);
        }

        ++num_samples;
    }

    return num_samples;
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

    std::vector<float> x_data;
    std::vector<float> t_data;
    std::vector<float> x_test_data;
    std::vector<float> t_test_data;
    size_t n_train_samples = load_mnist_csv(train_path.string(), x_data, t_data);
    size_t n_test_samples = load_mnist_csv(test_path.string(), x_test_data, t_test_data);


    std::println("Loaded {} train samples, {} test samples",
                 n_train_samples, n_test_samples);

    // Build 2 layer model: 784 -> 128 -> ReLU -> 10
    Sequential model(
        Linear(num_pixels, 128),
        ReLU(),
        Linear(128, num_classes)
    );

    float lr = 0.05f;
    size_t epochs = 5;          // Number of times the model is trained over whole train set (repeat)
    size_t batch_size = 5;

    std::mt19937 gen(std::random_device{}());
    BatchMaker batcher(n_train_samples);
    // Training

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        float epoch_loss = 0.0f;
        batcher.shuffle(gen);

        for (size_t s = 0; s < n_train_samples; s += batch_size) {
            size_t current_bs = std::min(batch_size, n_train_samples - s);
            TensorView x_batch = batcher.x_batch(x_data, num_pixels, s, current_bs);
            TensorView t_batch = batcher.t_batch(t_data, num_classes, s, current_bs);
            TensorView y_batch = model.pred(x_batch);
            model.grad_loss(y_batch, t_batch);
            model.backward();
            model.step(lr, current_bs);

            // End of core training loop

            float loss = total_mse_loss(y_batch, t_batch);
            epoch_loss += loss;
            if ((s / batch_size + 1) % 10000 == 0) {
                std::chrono::steady_clock::time_point cur_time = std::chrono::steady_clock::now();
                std::println("Epoch {} step {}/{} - running avg loss = {} t = {}",
                             epoch, s + 1, n_train_samples,
                             epoch_loss / static_cast<float>(s + 1),
                             std::chrono::duration_cast<std::chrono::seconds> (cur_time - begin).count());
            }
        }

        float avg_loss = epoch_loss / static_cast<float>(n_train_samples);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::println("Epoch {} finished. Avg loss = {} runtime epoch = {}ms", epoch, avg_loss, std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count());
    }

    // Evaluation on test set
    int correct = 0;
    for (size_t j = 0; j < n_test_samples; j++) {
        TensorView x_batch = make_batch_view(x_test_data, num_pixels, j, 1);
        TensorView y_batch = model.pred(x_batch);

        const auto& yr = y_batch.data;
        int pred = 0;
        for (int i = 1; i < num_classes; ++i) {
            if (yr[i] > yr[pred]) pred = i;
        }

        if (t_test_data[j * 10 + pred]) ++correct;
    }

    float acc = 100.0f * static_cast<float>(correct) / static_cast<float>(n_test_samples);
    std::println("Test accuracy: {}/{} ({:.2f}%)",
                 correct, n_test_samples, acc);

    return 0;
}
