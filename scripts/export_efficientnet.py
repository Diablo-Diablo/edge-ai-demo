from pathlib import Path
import torch
import torchvision.models as models

# First attempt to utilize pathlib to manage paths and 
# directories. This will help in organizing the project 
# structure and ensuring that the necessary directories 
# exist before exporting the model.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)  # 防止 models/ 不存在

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = MODELS_DIR / "efficientnet_b0_fp32.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamo=False,   #这里强制使用内联合并权重
    do_constant_folding=True
)

print(f"✅ Exported FP32 ONNX to: {onnx_path}")