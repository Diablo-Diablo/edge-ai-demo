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


## 📂 Project Structure
edge-ai-demo/

├── train.py                # PyTorch training script

├── export_onnx.py          # Convert .pth → .onnxx

├── quantize.py             # Post-training quantization (PyTorch / ONNX / TFLite)

├── infer_onnx.py           # ONNX Runtime inference demo

├── mobilenetv2.pth         # Trained FP32 model

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
- [ ] Quantization (INT8)
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
python quantize.py

## 📊 Results(To Be Updated)
| Model | Size (MB) | Top-1 Acc (%) | CPU Inference (ms) |
|-------|-----------|---------------|--------------------|
| FP32  | ~14       | xx.x          | xx                 |
| INT8  | ~3.5      | xx.x          | xx                 |

*(Fill in after running)*

## 📚 References
- PyTorch ONNX Export: https://pytorch.org/docs/stable/onnx.html
- ONNX Runtime Inference: https://onnxruntime.ai/
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- TensorFlow Lite Converter

## 🙋 Author
**Su, Jinshi**  
CUHK MSc (EE) – Embedded AI / Edge Deployment Track  
Summer Self-study Project · 2026