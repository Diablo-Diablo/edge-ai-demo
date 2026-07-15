import onnxruntime as ort
import numpy as np

# 1. 加载 ONNX 模型
session = ort.InferenceSession("tiny_model.onnx")

# 2. 获取输入输出名字
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 3. 构造输入数据（batch_size=2, feature_dim=3）
x = np.random.randn(2, 3).astype(np.float32)

# 4. 推理
y = session.run([output_name], {input_name: x})

# 5. 打印结果
print("Input shape:", x.shape)
print("Output:\n", y[0])