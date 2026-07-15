import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)

model = TinyNet()
model.eval()

dummy_input = torch.randn(1, 3)
torch.onnx.export(
    model,
    dummy_input,
    "tiny_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

print("✅ ONNX exported")