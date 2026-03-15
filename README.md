# Handwritten-Digit-Recognition-YOLOv8
基于 YOLOv8 分类模型 + Tkinter 可视化界面的手写数字识别工具，内置一键训练、实时手写识别、GPU加速、自动数字分割等核心功能，开箱即用。

## 功能亮点
- ✅ **GPU/CPU自适应**：自动检测CUDA设备，优先启用GPU加速，无GPU自动切换CPU运行
- ✅ **智能早停训练**：训练时连续5轮loss不下降自动终止，避免过拟合，支持Ctrl+C安全中断并保存模型
- ✅ **开箱即用的GUI界面**：自带手写画布，支持鼠标手写、一键识别、一键清空，操作零门槛
- ✅ **专业级图像预处理**：内置Otsu自动二值化、去噪、等比缩放、居中对齐、多数字竖直投影分割
- ✅ **多数字连续识别**：支持画布上书写多个数字，自动分割并逐个输出识别结果+置信度
- ✅ **零额外数据准备**：训练自动下载MNIST数据集，无需手动上传/准备数据文件

## 技术栈
- **核心框架**：PyTorch / Ultralytics YOLOv8
- **视觉处理**：Pillow (PIL), NumPy
- **交互界面**：Tkinter
- **工程化**：Git 版本管理、自动早停、GPU 内存优化

## 项目结构
Handwritten-Digit-Recognition-YOLOv8/├── main.py # 核心主程序（训练 + 推理 GUI）├── README.md # 项目说明文档├── requirements.txt # 项目依赖清单└── .gitignore # Git 忽略规则
plaintext

## 核心代码逻辑
### 1. 训练模块 (`train_simple_model`)
- 自动检测已有模型权重，避免重复训练
- 配置 `weight_decay` 与低学习率 (`lr0=0.001`) 保证训练稳定性
- 训练前自动清理 GPU 缓存，提升数据传输效率

### 2. 图像预处理模块 (`preprocess_image`)
- **核心亮点**：手动实现 Otsu 算法寻找最优分割阈值
- 将输入图像统一处理为 MNIST 风格的 32x32 三通道输入
- 自动提取数字边界框并居中，提升识别鲁棒性

### 3. 多数字分割模块 (`_segment_digits`)
- 基于列像素统计的竖直投影法
- 自动检测数字间隙，实现多数字连续书写与识别
- 分割时自动扩展边界，避免切断笔画

## 环境要求
- Python 3.8 ~ 3.11
- 支持Windows/macOS/Linux
- 可选：NVIDIA显卡+CUDA环境（大幅提升训练速度）

## 快速开始
```bash
# 克隆仓库
git clone https://github.com/HH-ANTENNA/Handwritten-Digit-Recognition-YOLOv8.git
cd Handwritten-Digit-Recognition-YOLOv8

# 安装依赖
pip install -r requirements.txt

# 运行
python main.py