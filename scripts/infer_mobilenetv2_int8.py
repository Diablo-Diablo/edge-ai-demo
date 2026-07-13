import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import os

#  项目根目录（edge-ai-demo）
PROJECT_ROOT = r"C:\Users\dd4\edge-ai-demo"

# 拼接路径
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mobilenetv2_quant.onnx")
LABEL_PATH = os.path.join(PROJECT_ROOT, "imagenet_classes.txt")
IMAGE_PATH = os.path.join(PROJECT_ROOT, "countach.jpg")

# ---------- 1. 加载 INT8 模型 ----------
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("🖥️  Execution Provider:", session.get_providers())

# ---------- 2. 加载 ImageNet 标签 ----------
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# ---------- 3. 图像预处理（必须与训练/导出时一致） ----------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).numpy()  #输入数据形状

# ---------- 4. 推理 + 计时 ----------
start = time.perf_counter()
logits = session.run([output_name], {input_name: input_tensor})[0]
end = time.perf_counter()

# ---------- 5. 解析结果 ----------
idx = np.argmax(logits)
confidence = float(np.max(logits))

print("🖼️  Predicted class :", labels[idx])
print("🔢  Class index    :", idx)
print("📈  Confidence     :", round(confidence, 4))
print(f"⏱️  Inference time : {(end - start) * 1000:.2f} ms")