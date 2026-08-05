#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {
constexpr int kResizeSize = 256;
constexpr int kCropSize = 224;
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

std::vector<std::string> LoadLabels(const std::string& label_path) {
    std::ifstream file(label_path);
    if (!file) {
        throw std::runtime_error("failed to open labels file: " + label_path);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        labels.push_back(line);
    }
    return labels;
}

std::vector<float> PreprocessImage(const std::string& image_path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* image = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!image) {
        throw std::runtime_error("failed to load image: " + image_path);
    }

    const int new_width = (width < height) ? kResizeSize : static_cast<int>(std::round(width * (kResizeSize / static_cast<float>(height))));
    const int new_height = (width < height) ? static_cast<int>(std::round(height * (kResizeSize / static_cast<float>(width)))) : kResizeSize;

    std::vector<float> resized(3 * new_width * new_height);
    for (int y = 0; y < new_height; ++y) {
        const float src_y = (new_height == 1) ? 0.0f : y * (height - 1.0f) / (new_height - 1.0f);
        const int y0 = static_cast<int>(src_y);
        const int y1 = std::min(y0 + 1, height - 1);
        const float wy = src_y - y0;
        for (int x = 0; x < new_width; ++x) {
            const float src_x = (new_width == 1) ? 0.0f : x * (width - 1.0f) / (new_width - 1.0f);
            const int x0 = static_cast<int>(src_x);
            const int x1 = std::min(x0 + 1, width - 1);
            const float wx = src_x - x0;

            for (int c = 0; c < 3; ++c) {
                const float p00 = image[(y0 * width + x0) * 3 + c] / 255.0f;
                const float p01 = image[(y0 * width + x1) * 3 + c] / 255.0f;
                const float p10 = image[(y1 * width + x0) * 3 + c] / 255.0f;
                const float p11 = image[(y1 * width + x1) * 3 + c] / 255.0f;
                const float top = p00 * (1.0f - wx) + p01 * wx;
                const float bottom = p10 * (1.0f - wx) + p11 * wx;
                resized[c * new_width * new_height + y * new_width + x] = top * (1.0f - wy) + bottom * wy;
            }
        }
    }

    stbi_image_free(image);

    const int crop_x = (new_width - kCropSize) / 2;
    const int crop_y = (new_height - kCropSize) / 2;
    std::vector<float> output(3 * kCropSize * kCropSize);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < kCropSize; ++y) {
            for (int x = 0; x < kCropSize; ++x) {
                const int src_x = crop_x + x;
                const int src_y = crop_y + y;
                float value = resized[c * new_width * new_height + src_y * new_width + src_x];
                value = (value - kMean[c]) / kStd[c];
                output[c * kCropSize * kCropSize + y * kCropSize + x] = value;
            }
        }
    }

    return output;
}
}  // namespace

int main() {
    try {
        const std::string project_root = "D:\\edge-ai-demo";
        const std::string model_path = project_root + "\\models\\mobilenetv2.onnx";
        const std::string label_path = project_root + "\\imagenet_classes.txt";
        const std::string image_path = project_root + "\\countach.jpg";

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mobilenet-infer");
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(4);
        session_opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        std::basic_string<ORTCHAR_T> model_path_ort(model_path.begin(), model_path.end());
        Ort::Session session(env, model_path_ort.c_str(), session_opts);

        const auto labels = LoadLabels(label_path);
        const auto input_data = PreprocessImage(image_path);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();
        const char* output_name = output_name_ptr.get();

        std::vector<int64_t> input_shape{1, 3, kCropSize, kCropSize};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_shape.data(),
            input_shape.size());

        auto start = std::chrono::steady_clock::now();
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            &input_name, &input_tensor, 1,
            &output_name, 1);
        auto end = std::chrono::steady_clock::now();

        float* logits = outputs[0].GetTensorMutableData<float>();
        size_t num_classes = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        size_t idx = static_cast<size_t>(std::distance(logits, std::max_element(logits, logits + num_classes)));
        float confidence = logits[idx];
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Predicted class: " << (idx < labels.size() ? labels[idx] : std::string("<label missing>")) << std::endl;
        std::cout << "Class index: " << idx << std::endl;
        std::cout << "Confidence (logit): " << confidence << std::endl;
        std::cout << "Inference time: " << ms << " ms" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
// 已修改
