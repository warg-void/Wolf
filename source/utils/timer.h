#pragma once

#include <chrono>
#include <print>

enum TimeUnit {Seconds, Milliseconds};

namespace wolf {
    class Timer {
        size_t call_count;
        double total_ms;
        size_t print_freq;
        TimeUnit unit;
        std::chrono::steady_clock::time_point begin;
        std::chrono::steady_clock::time_point end;
        uint64_t elapsed;
        int count;
        std::string msg;
    public:
        Timer(TimeUnit unit, size_t print_freq = 1, std::string msg = "") : print_freq(print_freq),
         unit(unit), elapsed(0), count(0), msg(msg) {}
        void start() {
            begin = std::chrono::steady_clock::now();       
        }
        void stop() {
            end = std::chrono::steady_clock::now();
            elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            count += 1;
            if (count % print_freq == 0) {
                auto ms = elapsed/1000000; 
                std::println("{} Ran {} times, t = {}ms", msg, count, ms);
            }
        }
    };
}