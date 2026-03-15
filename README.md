# Handwritten-Digit-Recognition-YOLOv8
✨ 基于 YOLOv8 分类模型 + Tkinter 可视化界面的端到端手写数字识别工具，内置一键训练、实时手写识别、GPU加速、自动数字分割等核心功能，开箱即用。

![GitHub Stars](https://img.shields.io/github/stars/HH-ANTENNA/Handwritten-Digit-Recognition-YOLOv8?style=flat-square)
![GitHub Forks](https://img.shields.io/github/forks/HH-ANTENNA/Handwritten-Digit-Recognition-YOLOv8?style=flat-square)
![License](https://img.shields.io/github/license/HH-ANTENNA/Handwritten-Digit-Recognition-YOLOv8?style=flat-square)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red?style=flat-square&logo=pytorch)

---

## 🎯 功能亮点
| 功能 | 描述 |
|------|------|
| 🚀 **GPU/CPU自适应** | 自动检测CUDA设备，优先启用GPU加速，无GPU自动切换CPU运行 |
| ⏱️ **智能早停训练** | 训练时连续5轮loss不下降自动终止，避免过拟合，支持Ctrl+C安全中断并保存模型 |
| 🖼️ **开箱即用的GUI** | 自带手写画布，支持鼠标手写、一键识别、一键清空，操作零门槛 |
| 🎨 **专业图像预处理** | 内置Otsu自动二值化、去噪、等比缩放、居中对齐、多数字竖直投影分割 |
| 🔢 **多数字连续识别** | 支持画布上书写多个数字，自动分割并逐个输出识别结果+置信度 |
| 📦 **零额外数据准备** | 训练自动下载MNIST数据集，无需手动上传/准备数据文件 |

---

## 🛠️ 技术栈
| 技术 | 用途 | 版本要求 |
|------|------|----------|
| PyTorch / Ultralytics YOLOv8 | 核心深度学习框架与分类模型 | ≥1.9.0 / ≥8.0.0 |
| Pillow (PIL) | 图像加载与预处理 | ≥8.0.0 |
| NumPy | 数值计算与像素统计 | ≥1.21.0 |
| Tkinter | GUI可视化交互界面 | Python内置（无需额外安装） |
| Git | 版本管理与工程化 | 任意版本 |

---

## 📂 项目结构
```text
Handwritten-Digit-Recognition-YOLOv8/
|-- main.py                 # 核心主程序（训练 + 推理GUI全逻辑）
|-- README.md               # 项目说明文档（本文件）
|-- requirements.txt        # 项目依赖清单
|-- .gitignore              # Git忽略规则（过滤模型/数据集/缓存）
```
## 📝 核心代码逻辑
<details>
<summary>点击展开查看详细实现</summary>

### 1. 训练模块 (`train_simple_model`)
- 自动检测已有模型权重，避免重复训练
- 配置 `weight_decay` 与低学习率 (`lr0=0.001`) 保证训练稳定性
- 训练前自动清理GPU缓存，提升数据传输效率
- 支持Ctrl+C安全中断，自动保存当前最优模型

### 2. 图像预处理模块 (`preprocess_image`)
- **核心亮点**：手动实现Otsu阈值算法，不依赖OpenCV，自适应处理不同光照的手写图像
- 将输入图像统一处理为MNIST风格的32x32三通道输入
- 自动提取数字边界框并居中，提升识别鲁棒性

### 3. 多数字分割模块 (`_segment_digits`)
- 基于列像素统计的竖直投影法，自动检测数字间隙
- 支持连续手写多数字的分割与识别
- 分割时自动扩展边界，避免切断笔画
</details>

---

## 📌 环境要求
- ✅ Python 3.8 ~ 3.11
- ✅ 支持 Windows/macOS/Linux
- ⚡ 可选：NVIDIA显卡 + CUDA环境（大幅提升训练速度）
##📄 许可证
-本项目采用 Apache-2.0 许可证，仅供学习交流与面试展示使用。
