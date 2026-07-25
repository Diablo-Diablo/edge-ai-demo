import torch
import torchvision.models as models

# 1. Load pre-trained EfficientNet-B0
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.eval()

# 2. Dummy input (Batch 1, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX
onnx_path = "models/efficientnet_b0_fp32.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=13,  # Using 13 for better ORT compatibility
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"✅ Exported FP32 ONNX to: {onnx_path}")