// Decode throughput: prefill 512 tokens, then time five 128-token suffixes and print tok/s.
// Build: nvcc -O3 -std=c++17 -arch=compute_80 -code=sm_86,compute_80 inference/cuda/bench.cu -o bench
// Run:   ./bench model.bin
#include "decode.cuh"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: bench WEIGHTS.bin\n"; return 1; }
    zlm::Decoder decoder(argv[1], 640);
    auto id = [](int i) { return (i * 7919 + 137) % zlm::V; };  // fixed IDs; speed does not depend on them
    for (int i = 0; i < 512; ++i) decoder.forward(id(i), false);
    for (int trial = 0; trial <= 5; ++trial) {
        decoder.length = 512;  // suffix writes never modify the populated prefix
        zlm::check(cudaDeviceSynchronize());
        auto start = std::chrono::steady_clock::now();
        for (int i = 512; i < 640; ++i) decoder.forward(id(i));
        zlm::check(cudaDeviceSynchronize());
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (trial) std::cout << 128 / seconds << " tok/s\n";  // trial 0 is the warmup
    }
}
