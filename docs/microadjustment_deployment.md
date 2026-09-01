# 分类模型微调
> 关联实验：EfficientNet-B0 人像误识别为牛仔帽/军服

---

## 核心逻辑=
预训练模型（如EfficientNet-B0）已经学会了「提取通用特征」（边缘/纹理/物体部件），微调就是**冻结这些特征提取层，只替换最后一层分类头**，用少量新数据教模型把「牛仔帽/军服」这类特征映射到你自己定义的新类别（比如"Lee Van Hoff"）。

---

## 最小操作步骤
### 1. 改分类头

python

import torchvision.models as models

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

原输出1000类，改为1000+N（N是新类别数，比如加2个人像类就写1002）

model.classifier = torch.nn.Linear(model.classifier.in_features, 1000 + N)

### 2. 冻结特征层（关键，避免破坏预训练特征）

python

冻住所有特征提取层参数

for param in model.parameters():

param.requires_grad = False

仅解冻新分类头，只训练这部分

for param in model.classifier.parameters():

param.requires_grad = True

### 3. 准备小数据集
- 新类别每类仅需10~50张图，尽量和推理场景一致（比如要识别Lee的舞台照，就别用证件照训练）
- 预处理完全复用原模型配置：`Resize(256) → CenterCrop(224) → Normalize(ImageNet均值方差)`

### 4. 训练
- 损失函数：`torch.nn.CrossEntropyLoss()`
- 优化器：Adam，学习率`1e-3`
- 训练轮次：10~20 epoch即可，不用训太久

### 5. 导出+部署
- 导出ONNX：复用现有的 `scripts/export_efficientnet.py`，仅改输出路径
- 量化：直接用现有 `quantize_static.py`，参数不用改
- 推理：直接用现有 `infer_efficientnet.py`，仅改模型路径

---

## 避坑点
 不要一开始就全量微调：数据少的时候全量微调会破坏预训练特征，精度反而下降，先冻特征层只训分类头
 新数据要覆盖场景：比如要识别Lee的不同角度、不同光照照片，训练集里就要包含这些，否则推理泛化差
 量化流程完全不变：微调后的模型和原模型量化逻辑100%一致，不用改任何参数

---

## 回头捡起点
1. 先看本笔记回忆核心逻辑
2. 翻 `scripts/export_efficientnet.py` 确认路径配置
3. 准备新类别图片，按步骤训练即可
---
> 当前优先级：先跑完EfficientNet-B0的FP32/INT8 benchmark，本笔记仅作存档，需要时再翻。