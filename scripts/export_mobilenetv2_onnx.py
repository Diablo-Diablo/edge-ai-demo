import torch
import torchvision.models as models

# 1. 加载预训练 MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.eval()

# 2. 构造示例输入（ImageNet 标准尺寸）
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv2.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    },
    opset_version=12
)

print("✅ mobilenetv2.onnx exported")
