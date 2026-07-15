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

├── train.py                # PyTorch training script

├── export_onnx.py          # Convert .pth → .onnxx

├── quantize.py             # Post-training quantization (PyTorch / ONNX / TFLite)

├── infer_onnx.py           # ONNX Runtime inference demo

├──export_mobilenetv2_onnx.py       # Trained FP32 model

├── mobilenetv2.onnx        # Exported ONNX model

├── mobilenetv2_quant.onnx  # Quantized ONNX model (INT8)

├── requirements.txt

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

## 📊 Results(To Be Updated)

| Model | Input | Opset | Size (MB) | Top‑1 Prediction | Logit | Inference Time (ms)[100 times,AVG] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MobileNetV2 (FP32) | `countach.jpg` | 18 (Dynamo) | 13.7 | sports car (817) | 16.5654 | 2.731 (ORT CPU) |
| MobileNetV2_Dyn (INT8) | `countach.jpg` | 18 (Dynamo) | 3.61 | sports car (817) | 15.8329 | 23.852 (ORT CPU) |
| MobileNetV2_Stat (INT8) | `countach.jpg` | 18 (Dynamo) | 3.58 | sports car (817) | 14.944 | 2.763 (ORT CPU) |

>Note：Dynamic quantization results in slightly higher latency on this CPU / batch size combination.

> This is due to runtime activation quantization overhead and the lightweight nature of MobileNetV2.

> INT8 quantization fuses weights and activations into INT8, eliminating runtime conversion overhead. This yields near-FP32 latency (2.76 ms) while reducing model size by 74%.


## 📚 References
- PyTorch ONNX Export: https://pytorch.org/docs/stable/onnx.html
- ONNX Runtime Inference: https://onnxruntime.ai/
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- TensorFlow Lite Converter

## 🙋 Author
**Su, Jinshi**  
CUHK MSc (EE) – Embedded AI / Edge Deployment Track  
Summer Self-study Project · 2026