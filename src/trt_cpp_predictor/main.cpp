#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

// A simple logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <input_batch.bin>" << std::endl;
        return -1;
    }

    const std::string engine_path = argv[1];
    const std::string batch_path = argv[2];

    Logger logger;

    // --- 1. Load Engine ---
    std::vector<char> engine_blob;
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (engine_file.good()) {
        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        engine_blob.resize(size);
        engine_file.read(engine_blob.data(), size);
    } else {
        std::cerr << "Error opening engine file: " << engine_path << std::endl;
        return -1;
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_blob.data(), engine_blob.size());
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // --- 2. Load Input Data ---
    std::vector<float> host_input;
    std::ifstream input_file(batch_path, std::ios::binary);
    if (input_file.good()) {
        input_file.seekg(0, std::ios::end);
        size_t size = input_file.tellg();
        input_file.seekg(0, std::ios::beg);
        host_input.resize(size / sizeof(float));
        input_file.read(reinterpret_cast<char*>(host_input.data()), size);
    } else {
        std::cerr << "Error opening input batch file: " << batch_path << std::endl;
        return -1;
    }

    // --- 3. Allocate Buffers ---
    void* buffers[5]; // 1 input + 4 outputs
    const int input_idx = 0; // Assuming single input
    cudaMalloc(&buffers[input_idx], host_input.size() * sizeof(float));

    // Get output buffer sizes from the engine
    std::vector<int> host_num_dets(1);
    std::vector<float> host_bboxes(100 * 4); // Assuming max 100 detections (topk)
    std::vector<float> host_scores(100);
    std::vector<int> host_labels(100);

    cudaMalloc(&buffers[1], host_num_dets.size() * sizeof(int));
    cudaMalloc(&buffers[2], host_bboxes.size() * sizeof(float));
    cudaMalloc(&buffers[3], host_scores.size() * sizeof(float));
    cudaMalloc(&buffers[4], host_labels.size() * sizeof(int));

    // --- 4. Run Inference ---
    auto start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(buffers[input_idx], host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    context->executeV2(buffers);
    cudaMemcpy(host_num_dets.data(), buffers[1], host_num_dets.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_bboxes.data(), buffers[2], host_bboxes.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_scores.data(), buffers[3], host_scores.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_labels.data(), buffers[4], host_labels.size() * sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // --- 5. Print Results ---
    std::cout << "Inference complete in " << elapsed.count() << " ms." << std::endl;
    int num_detections = host_num_dets[0];
    std::cout << "Found " << num_detections << " detections." << std::endl;

    for (int i = 0; i < num_detections; ++i) {
        std::cout << "  - Box: ["
                  << host_bboxes[i * 4 + 0] << ", " << host_bboxes[i * 4 + 1] << ", "
                  << host_bboxes[i * 4 + 2] << ", " << host_bboxes[i * 4 + 3] << "], "
                  << "Score: " << host_scores[i] << std::endl;
    }

    // --- 6. Cleanup ---
    for (int i = 0; i < 5; ++i) cudaFree(buffers[i]);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}