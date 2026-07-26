from pathlib import Path
import onnx
import os
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    QuantType,
    QuantFormat,
    CalibrationDataReader
)
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# ===== 路径配置（复用pathlib逻辑）=====
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT.parent / "models"
CALIB_DIR = PROJECT_ROOT.parent / "calib_images"

FP32_MODEL = MODELS_DIR / "efficientnet_b0_fp32.onnx"
STATIC_INT8_MODEL = MODELS_DIR / "efficientnet_b0_static_int8.onnx"
DYNAMIC_INT8_MODEL = MODELS_DIR / "efficientnet_b0_dynamic_int8.onnx"

onnx_model = onnx.load(FP32_MODEL)
input_name = onnx_model.graph.input[0].name

# ===== CalibrationDataReader（完全不用改，只要输入是224x224）=====
class MyCalibrationDataReader(CalibrationDataReader):
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
    
reader = MyCalibrationDataReader()

# ===== 静态量化核心代码（加1个必填参数，你之前踩过的坑）=====
def quantize_static_efnet():
  #  calib_reader = MyCalibrationDataReader(
   #     PROJECT_ROOT.parent / "calib_images"  # 之前的校准数据目录
   # )
    
    quantize_static(
    model_input=FP32_MODEL,
    model_output=STATIC_INT8_MODEL,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,   #需要手动设置at与wt匹配
    weight_type=QuantType.QUInt8
)
    print(f"✅ 静态INT8模型导出至: {STATIC_INT8_MODEL}")

def quantize_dynamic_efnet():
    quantize_dynamic(
        model_input=str(FP32_MODEL),
        model_output=str(DYNAMIC_INT8_MODEL),
        weight_type=QuantType.QUInt8,
        # 动态量化不需要校准数据，也不需要activation_type参数
    )
    print(f"✅ 动态INT8模型导出至: {DYNAMIC_INT8_MODEL}")

if __name__ == "__main__":
    quantize_static_efnet()
    #quantize_dynamic_efnet()