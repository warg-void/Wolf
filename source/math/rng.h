#pragma once
#include <random>

namespace wolf {
    struct RNG {
        std::mt19937 gen;
        RNG() : gen(std::random_device{}()) {}
    };
    
    inline RNG& rng() {
        static RNG instance;
        return instance;
    }
}