# Edge AI Deployment Demo – MobileNetV2 (PyTorch → ONNX → Quantization)

## 📌 Overview
This project demonstrates a basic **Edge AI deployment pipeline**:
- Train MobileNetV2 on CIFAR-10 / ImageNet-mini
- Export to ONNX format
- Apply Post-Training Quantization (PTQ)
- Run inference with ONNX Runtime

Target platforms: **Laptop (CPU) → Raspberry Pi → Embedded MCU**

## 🛠 Environment
- Python 3.9+
- PyTorch >= 1.13
- onnx / onnxruntime
- torchvision
- (Optional) tensorflow / tflite-runtime

Create environment:[bash]
conda create -n edgeai python=3.9
conda activate edgeai
pip install -r requirements.txt


## 📂 Project Structure (To Be Perfected)

edge-ai-demo/
├── models/
│   ├── mobilenetv2.onnx               #FP32
│   ├── mobilenetv2_quant.onnx         #Dynamic
│   └── mobilenetv2_quant_static.onnx  #Static
├── scripts/
│   ├── export_onnx.py
│   ├── quantize_dynamic.py            
│   ├── quantize_static.py              
│   └── benchmark.py                   #benchmark,100 times 
├── docs/
│   └── quantization_notes.md          
├── assets/
│   └── benchmark_results.png          
├── countach.jpg
├── imagenet_classes.txt
└── README.md

## Project Status
- [x] Conda env `edgeai` setup 
- [x] PyTorch CPU sanity check passed
- [x] ONNX export (`export_onnx.py`)
- [x] ONNX Runtime inference (`infer_onnx.py`)
- [x] Netron Visualization 
- [x] Quantization(dynamic) (INT8)
- [ ] Quantization(static) (INT8)
- [ ] Deployment benchmark (Raspberry Pi / Laptop)

## 🚀 Quick Start

### 1. Train model  [bash]
python train.py

### 2. Export to ONNX  [bash]
python export_onnx.py

### 3. Visualize model  [bash]
pip install netron 

netron mobilenetv2.onnx

### 4. Run inference  [bash]
python infer_onnx.py

### 5. Quantization (WIP)  [bash]
python quantize_mobilenetv2.py
python static_quantize_mobilenetv2.py

### 6.Deployment
>Deployment on edge devices is straightforward:

>the statically quantized ONNX model is self-contained, requiring only ONNX Runtime (CPU) to run—no re-quantization or platform-specific tooling on the target device.

## 📊 Results(To Be Updated)

### Quantization
| Model | Input | Opset | Size (MB) | Top‑1 Prediction | Logit | Inference Time (ms)[100 times,AVG] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MobileNetV2 (FP32) | `countach.jpg` | 18 (Dynamo) | 13.7 | sports car (817) | 16.5654 | 2.731 (ORT CPU) |
| MobileNetV2_Dyn (INT8) | `countach.jpg` | 18 (Dynamo) | 3.61 | sports car (817) | 15.8329 | 23.852 (ORT CPU) |
| MobileNetV2_Stat (INT8) | `countach.jpg` | 18 (Dynamo) | 3.58 | sports car (817) | 14.944 | 2.763 (ORT CPU) |

> ✅ 74% smaller model  
> ✅ Near-FP32 inference speed  
> ✅ No runtime quantization overhead  

See [`docs/quantization_notes.md`](docs/quantization_notes.md) for details.



## 📚 References
- PyTorch ONNX Export: https://pytorch.org/docs/stable/onnx.html
- ONNX Runtime Inference: https://onnxruntime.ai/
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- TensorFlow Lite Converter

## 🙋 Author
**Su, Jinshi**  
CUHK MSc (EE) – Embedded AI / Edge Deployment Track  
Summer Self-study Project · 2026