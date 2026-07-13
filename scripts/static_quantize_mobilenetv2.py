import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, CalibrationMethod
import os
import glob

# ---------- 1. 输入/输出路径 ----------
input_model = "models/mobilenetv2_.onnx"
output_model = "models/mobilenetv2_quant.onnx"

# ---------- 2. 准备校准数据----------(为静态量化做准备)
# 用现有的 countach.jpg + 复制几份模拟多张图
# 实际项目中用 10~50 张真实图片
calibration_images = []
os.makedirs("calib_images", exist_ok=True)

# 复制 countach.jpg 为 10 张校准图
for i in range(10):
    img_path = f"calib_images/img_{i}.jpg"
    if not os.path.exists(img_path):
        os.system(f"copy countach.jpg {img_path}")  # Windows
        # Linux/Mac: os.system(f"cp countach.jpg {img_path}")

calibration_images = glob.glob("calib_images/*.jpg")

print(f"📸 校准图片数量: {len(calibration_images)}")

# ---------- 3. 动态量化（推荐新手）----------
# 动态量化：只量化权重，激活值运行时动态量化 → 快且稳
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QUInt8,           # 量化权重为 UINT8
    activation_type=QuantType.QUInt8,        # 激活值也量化
    calibrate_method=CalibrationMethod.MinMax,  # 最简单校准
    nodes_to_exclude=[],                     # 不排除任何节点
    extra_options={
        'EnableSubgraph': False,
        'ActivationSymmetric': False,
        'WeightSymmetric': False,
        'ForceQuantizeNoInputCheck': False,
    }
)

print(f"✅ 量化完成！输出模型: {output_model}")
print(f"📦 原模型大小: {os.path.getsize(input_model) / 1024 / 1024:.2f} MB")
print(f"📦 量化后大小: {os.path.getsize(output_model) / 1024 / 1024:.2f} MB")