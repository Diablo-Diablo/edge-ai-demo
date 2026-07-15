import onnx
import os
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader
)
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# ---------- 强制 CPU 量化环境 ----------
os.environ["ORT_QUANTIZATION_DEBUG"] = "1"

MODEL_PATH = "C:\\Users\\dd4\\edge-ai-demo\\models\\mobilenetv2.onnx"
OUT_MODEL = "C:\\Users\\dd4\\edge-ai-demo\\models\\mobilenetv2_quant_static.onnx"
CALIB_DIR = "C:\\Users\\dd4\\edge-ai-demo\\calib_images"

onnx_model = onnx.load(MODEL_PATH)
input_name = onnx_model.graph.input[0].name

class MyCalibReader(CalibrationDataReader):
    def __init__(self):
        self.paths = [
            os.path.join(CALIB_DIR, f)
            for f in os.listdir(CALIB_DIR)
            if f.endswith(".jpg")
        ]
        self.idx = 0
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def get_next(self):
        if self.idx >= len(self.paths):
            return None
        img = Image.open(self.paths[self.idx]).convert("RGB")
        x = self.tf(img).unsqueeze(0).numpy()
        self.idx += 1
        return {input_name: x}

    def rewind(self):
        self.idx = 0

reader = MyCalibReader()

# ---------- 关键：只保留 weight_type，不写 activation_type ----------
quantize_static(
    model_input=MODEL_PATH,
    model_output=OUT_MODEL,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,   #需要手动设置at与wt匹配
    weight_type=QuantType.QUInt8
)

print("✅ 静态量化完成")
print(f"✅ 量化模型已保存至: {OUT_MODEL}")