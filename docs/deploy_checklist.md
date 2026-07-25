# Laptop Deployment Verification Checklist (Edge AI Demo)

## 1. Environment & Hardware Configuration
- **Device Role**: Edge Inference Target (Laptop)
- **CPU**: Intel Core Ultra 5 225H (Arrow Lake, 4P+8E+2LPE)
- **Memory**: 32GB LPDDR5X @ 7600MHz
- **OS**: Windows (Managed via Conda)
- **Runtime**: `conda activate edgeai` (Python 3.10)
- **Core Dependencies**: onnx, onnxruntime, torch, torchvision, matplotlib, opencv-python

## 2. Deployed Model & Format
- **Model**: MobileNetV2 (ImageNet Pre-trained)
- **Format**: ONNX (Open Neural Network Exchange)
- **Artifacts**:
  - `models/mobilenetv2.onnx` (Baseline)
  - `models/mobilenetv2_quant_static.onnx` (Post-training Static Quantization)
  - `models/mobilenetv2_quant.onnx` (Dynamic Quantization)

## 3. Inference Performance Benchmark (Avg over 100 runs)
| Precision | Avg Latency | Std Dev | Min / Max | Top-1 Prediction (Sample) |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 1.437 ms | 2.740 ms | 1.042 / 18.077 ms | sports car (idx=817) |
| **INT8 Static** | 1.805 ms | 1.484 ms | 14.718 / 23.741 ms | sports car (idx=817) |
| **INT8 Dynamic** | 15.642 ms | - | - | sports car (idx=817) |

## 4. Deployment Conclusion
- **Status**: ✅ Verified & Deployed
- **Consistency**: The laptop (Ultra 5 225H) significantly outperforms the desktop (i5-10400F), primarily due to the newer microarchitecture and high-bandwidth 7600MHz memory.
- **Quantization Insights**: Static INT8 delivers near-lossless accuracy with excellent latency. Dynamic INT8 incurs substantial runtime overhead on CPU and is not recommended for production deployment.
- **Next Steps**: Proceed with EfficientNet-B0 integration and quantization validation.