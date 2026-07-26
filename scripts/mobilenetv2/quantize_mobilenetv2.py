import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, CalibrationMethod
import os
import glob

# ---------- 1. 输入/输出路径 ----------
input_model = "models/mobilenetv2.onnx"
output_model = "models/mobilenetv2_quant.onnx"


# ---------- 2. 动态量化----------
# 动态量化：只量化权重，激活值运行时动态量化 → 快且稳
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QUInt8,           # 量化权重为 UINT8    
    nodes_to_exclude=[],                     # 不排除任何节点
    extra_options={
        'EnableSubgraph': False,
        'ActivationSymmetric': False,
        'WeightSymmetric': False,
        'ForceQuantizeNoInputCheck': False,
    },
    per_channel=True,      # 可选：按通道量化，精度更好
    reduce_range=False     # 可选：视 CPU VNNI 支持而定
)



print(f"✅ 量化完成！输出模型: {output_model}")
print(f"📦 原模型大小: {os.path.getsize(input_model) / 1024 / 1024:.2f} MB")
print(f"📦 量化后大小: {os.path.getsize(output_model) / 1024 / 1024:.2f} MB")