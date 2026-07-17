# ONNX Runtime INT8 Quantization on Windows CPU

## Environment
- CPU: Intel i5-10400F
- ONNX Runtime: 1.23.2
- Execution Provider: CPUExecutionProvider

## Models Compared

| Model | Input | Opset | Size (MB) | Top‑1 Prediction | Logit | Inference Time (ms)[100 times,AVG] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MobileNetV2 (FP32) | `countach.jpg` | 18 (Dynamo) | 13.7 | sports car (817) | 16.5654 | 2.731 (ORT CPU) |
| MobileNetV2_Dyn (INT8) | `countach.jpg` | 18 (Dynamo) | 3.61 | sports car (817) | 15.8329 | 23.852 (ORT CPU) |
| MobileNetV2_Stat (INT8) | `countach.jpg` | 18 (Dynamo) | 3.58 | sports car (817) | 14.944 | 2.763 (ORT CPU) |

## Key Findings

### 1. Dynamic Quantization is Slow on CPU
Dynamic quantization introduces runtime Dequantize/Quantize nodes.
On CPU, this overhead dominates the compute savings, resulting in **~9x slowdown**.

### 2. Static Quantization Requires Explicit Activation Type
In ORT 1.23.2 Windows CPU builds, `activation_type` must be manually set:

```python
quantize_static(
    ...,
    activation_type=QuantType.QUInt8,  # Required, not inferred
    weight_type=QuantType.QUInt8,
)
```

Failure to do so results in:
```
ValueError: ONNXRuntime quantization doesn't support data format: activation_type=QuantType.QInt8
```

### 3. Why Static Quantization is Fast
Static quantization fuses weights and activations into INT8 **at export time**.
This eliminates runtime conversion overhead, achieving near-FP32 latency with 74% model size reduction.

## Deployment Implication
The statically quantized `.onnx` is **self-contained**:
- No re-quantization on target device
- Only `onnxruntime` (CPU) required
- Runs identically on x86 and ARM64 edge devices

## How to Reproduce
```bash
python scripts/static_quantize_mobilenetv2.py
python scripts/benchmark_inference.py 
```