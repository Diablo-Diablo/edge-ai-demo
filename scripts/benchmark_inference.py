import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import os

# ====================== 配置 ======================
os.environ["ORT_LOG_LEVEL"] = "ERROR"  # 隐藏 warning，让输出干净

PROJECT_ROOT = "C:\\Users\\dd4\\edge-ai-demo"  # 当前目录，或改为你的项目路径
IMAGE_PATH = os.path.join(PROJECT_ROOT, "countach.jpg")
LABEL_PATH = os.path.join(PROJECT_ROOT, "imagenet_classes.txt")

# 模型路径（根据你的实际情况修改）
MODELS = {
    "FP32": "models/mobilenetv2.onnx",
    "INT8_Static": "models/mobilenetv2_quant_static.onnx",
    "INT8_Dynamic": "models/mobilenetv2_quant.onnx",   # 如果有可取消注释
}

NUM_RUNS = 100          # 重复次数
WARMUP_RUNS = 10        # 预热次数（避免冷启动影响）

# ====================== 预处理 ======================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).numpy()

# ====================== 加载标签 ======================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# ====================== 基准测试函数 ======================
def benchmark_model(model_path, model_name):
    print(f"\n🚀 测试模型: {model_name} ({model_path})")
    
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Warmup
    print(f"   预热 {WARMUP_RUNS} 次...")
    for _ in range(WARMUP_RUNS):
        session.run([output_name], {input_name: input_tensor})
    
    # 正式测试
    times = []
    print(f"   正式推理 {NUM_RUNS} 次...")
    for i in range(NUM_RUNS):
        start = time.perf_counter()
        logits = session.run([output_name], {input_name: input_tensor})[0]
        end = time.perf_counter()
        times.append((end - start) * 1000)  # 转为 ms
    
    # 统计
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # 预测结果（取最后一次）
    idx = np.argmax(logits)
    confidence = float(np.max(logits))
    
    print(f"   ✅ 平均推理时间: {avg_time:.3f} ms")
    print(f"   📊 标准差: {std_time:.3f} ms")
    print(f"   ⏱️  最快/最慢: {min_time:.3f} / {max_time:.3f} ms")
    print(f"   🖼️  预测: {labels[idx]} (index={idx}, logit={confidence:.4f})")
    
    return avg_time

# ====================== 执行所有模型 ======================
results = {}
for name, path in MODELS.items():
    if os.path.exists(path):
        avg = benchmark_model(path, name)
        results[name] = avg
    else:
        print(f"⚠️  模型不存在: {path}")

# ====================== 最终对比 ======================
print("\n" + "="*60)
print("🏆 最终平均推理时间对比 (100次平均)")
print("="*60)
for name, avg in results.items():
    print(f"{name:15s}: {avg:6.3f} ms")
print("="*60)