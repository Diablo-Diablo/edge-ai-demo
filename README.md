# Edge AI Deployment Demo – MobileNetV2 (PyTorch → ONNX → Quantization)

## 📌 Overview
This project demonstrates a basic **Edge AI deployment pipeline**:
- Train MobileNetV2 on CIFAR-10 / ImageNet-mini
- Export to ONNX format
- Apply Post-Training Quantization (PTQ)
- Run inference with ONNX Runtime

## 🛠 Environment
- Python 3.9+
- PyTorch >= 1.13
- onnx / onnxruntime
- torchvision
- (Optional) tensorflow / tflite-runtime

## 📂 Project Structure
train.py        # PyTorch training script
export_onnx.py  # Convert .pth → .onnx
quantize.py     # Post-training quantization (PyTorch or TFLite)
infer_onnx.py   # ONNX Runtime inference demo
requirements.txt

## Project Status
- [x] Conda env `edgeai` setup 
- [x] PyTorch CPU sanity check passed
- [x] ONNX export (`export_onnx.py`)
- [x] ONNX Runtime inference (`infer_onnx.py`)
- [ ] Quantization (INT8)
- [ ] Deployment on Raspberry Pi / Laptop

## 🚀 Quick Start
bash
pip install -r requirements.txt
python train.py           # train model, saves mobilenetv2.pth
python export_onnx.py     # exports mobilenetv2.onnx
python quantize.py        # generates mobilenetv2_quant.onnx / .tflite
python infer_onnx.py      # run inference on sample image

## 📊 Results
| Model | Size (MB) | Top-1 Acc (%) | CPU Inference (ms) |
|-------|-----------|---------------|--------------------|
| FP32  | ~14       | xx.x          | xx                 |
| INT8  | ~3.5      | xx.x          | xx                 |

*(Fill in after running)*

## 📚 References
- PyTorch ONNX Export: https://pytorch.org/docs/stable/onnx.html
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- TensorFlow Lite Converter

## 🙋 Author
Su, Jinshi CUHK MSc(EE) – Embedded AI / Edge Deployment Track  
Summer Self-study Project 2026