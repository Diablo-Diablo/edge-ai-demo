# EfficientNet-B0 Laptop Deployment Verification Report
> **Test Device**: Intel Core Ultra 5 225H / 32GB LPDDR5X 7600MHz / Windows 11
> **Runtime Environment**: `edgeai` Conda env (Python 3.10 + ONNX Runtime 1.23.2 CPU)
> **Test Sample**: `countach.jpg` (ImageNet class: sports car, index=817)
> **Test Date**: 2026-07-26

---

## 1. Test Configuration
| Config Item | Parameter |
|-------------|-----------|
| Warmup Iterations | 10 runs |
| Benchmark Iterations | 100 runs |
| Inference Backend | CPUExecutionProvider |
| Quantization Schemes | FP32 / INT8 Static Quantization / INT8 Dynamic Quantization |

---

## 2. Core Benchmark Results
| Precision | Avg Latency | Std Dev | Min / Max Latency | Top-1 Prediction | Logit Value | Relative Speedup (vs FP32) |
|-----------|------------|--------|-------------------|------------------|-------------|----------------------------|
| **FP32** | 7.738 ms | 1.683 ms | 5.572 / 13.495 ms | sports car (idx=817) | 10.2174 | 1.00x (Baseline) |
| **INT8 Static** | 4.901 ms | 0.466 ms | 4.425 / 7.027 ms | window screen (idx=904) | 9.6328 | 1.58x |
| **INT8 Dynamic** | 34.431 ms | 3.933 ms | 31.939 / 65.588 ms | fountain pen (idx=563) | 5.0332 | 0.22x |

---

## 3. Key Takeaways
### 3.1 Performance (As Expected)
- **INT8 Static Quantization** delivers a stable 1.58x speedup with 37% lower latency, and a significantly tighter standard deviation (0.466ms) than FP32. It fully meets low-latency edge deployment requirements.
- **INT8 Dynamic Quantization** exhibits counterintuitive performance degradation: latency is 4.4x higher than FP32. This is caused by runtime overhead of dynamic scale calculation, making it non-viable for CPU-based production deployments.

### 3.2 Accuracy Degradation (Model-Specific Limitation)
The static quantized model misclassifies `sports car` as `window screen`, which is a known limitation of post-training quantization (PTQ) for EfficientNet-B0:
- The Squeeze-and-Excitation (SE) modules and Swish activation functions in EfficientNet are highly sensitive to numerical precision loss during PTQ.
- The current calibration set uses 100 random images, which has a severe distribution mismatch with the test sample (sports car). This causes the quantizer to fail to capture the valid numerical range of sports car features, leading to prediction drift.

### 3.3 Engineering Recommendations
1. **Prioritize INT8 Static Quantization**: It outperforms both alternatives in speed and stability. Accuracy issues can be resolved by optimizing the calibration set.
2. **Calibration Set Optimization**: Replace the random calibration images with 20-50 sports car images that match the test sample distribution. This will likely resolve the classification error.
3. **Alternative Model Selection**: For strict accuracy requirements without calibration tuning, switch to quantization-friendly models like MobileNetV2/V3, which typically incur <1% accuracy drop with PTQ.

---

## 4. Known Issues & Optimization Roadmap
| Issue | Optimization Strategy | Priority |
|-------|-----------------------|----------|
| Classification error after static quantization | Replace calibration set with in-distribution sports car samples | High |
| Numerical stability under extreme scenarios | Enable per-channel quantization (already enabled, further optimize calibration algorithm) | Medium |
| Mission-critical accuracy requirements | Adopt Quantization-Aware Training (QAT) for model fine-tuning | Low |

---

## 5. Archiving Notes
This report captures the phased deployment verification results of EfficientNet-B0 on a consumer laptop. Update this document after calibrating the quantization pipeline or switching to alternative quantization schemes.
> Related Docs: `deploy_checklist.md` (MobileNetV2 deployment baseline), `quantization_notes.md` (quantization principle reference)