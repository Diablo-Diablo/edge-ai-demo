#include <stdio.h>

// 模拟一个简单的推理函数（类似 model.forward()）
void inference(float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        // 假装这是一个经过训练的模型计算
        output[i] = input[i] * 0.6f + 0.2f;
    }
}

int main() {
    printf("=== Edge AI Demo (C Version) ===\n");

    float input[3] = {1.0f, 2.0f, 3.0f};
    float output[3];

    // 执行推理
    inference(input, output, 3);

    // 打印结果（类似 PyTorch 的 print(tensor)）
    printf("Inference Results:\n");
    for (int i = 0; i < 3; i++) {
        printf("Output[%d] = %.3f\n", i, output[i]);
    }

    return 0;
}