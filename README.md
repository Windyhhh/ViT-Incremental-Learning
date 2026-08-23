# 👁️ Vision Transformer 增量学习系统 | ViT Incremental Learning

> **在 CIFAR-100 上探索 Vision Transformer 的渐进式学习策略，50 epochs 全流程训练，诊断结果可视化。**
>
> *Exploring progressive learning strategies for Vision Transformer on CIFAR-100, with full 50-epoch training pipeline and diagnostic result visualization.*

---

## 📌 项目简介 | Overview

本项目基于 Vision Transformer (ViT) 架构，在 CIFAR-100 数据集上实现增量学习（Incremental Learning）。通过渐进式训练策略，模型在不断学习新类别的同时，尽可能保留对旧类别的识别能力，缓解灾难性遗忘问题。

This project implements incremental learning based on Vision Transformer (ViT) architecture on the CIFAR-100 dataset. Through progressive training strategies, the model learns new categories while preserving recognition ability for old categories, mitigating catastrophic forgetting.

---

## ✨ 核心特性 | Features

| 特性 | Feature | 说明 |
|------|---------|------|
| 🎯 ViT 架构 | Vision Transformer | 基于 Transformer 的图像分类骨干网络 |
| 📈 增量学习 | Incremental Learning | 渐进式学习新类别，缓解灾难性遗忘 |
| 🧪 CIFAR-100 | 100 类图像数据集 | 100 个细粒度类别，50000 训练 + 10000 测试 |
| ⏱️ 50 Epochs | 完整训练流程 | 50 轮完整训练，多种策略对比 |
| 📊 诊断结果 | Diagnostic Results | 训练过程诊断，结果可视化分析 |
| 🗂️ 多策略对比 | Multi-Strategy | best / comprehensive / progressive 三种结果对比 |

---

## 📂 项目结构 | Project Structure

```
ViT-Incremental-Learning/
├── main.py                          # 主程序入口
├── requirements.txt                 # Python 依赖
├── README.md                        # 项目说明
├── 项目结构整理总结.md              # 项目结构总结
├── data/
│   └── cifar-100-python/           # CIFAR-100 数据集
│       ├── train                    # 训练集 (148MB)
│       ├── test                     # 测试集 (30MB)
│       └── meta                     # 元数据
├── results_best/                    # 最佳策略结果
│   └── diagnostic_results.pt        # 诊断结果
├── results_comprehensive/           # 综合策略结果
│   └── diagnostic_results.pt        # 诊断结果
└── results_progressive/             # 渐进式策略结果
    └── results_*.json               # 多轮训练结果
```

---

## 🚀 快速开始 | Quick Start

### 环境要求 | Requirements

```bash
pip install -r requirements.txt
```

### 运行训练 | Run Training

```bash
python main.py
```

---

## 🔬 技术细节 | Technical Details

### 增量学习策略 | Incremental Learning Strategy

- **渐进式学习**：模型分阶段学习新类别，每阶段保留旧类别知识
- **知识蒸馏**：利用教师模型指导学生模型，保留旧类别特征
- **正则化约束**：通过权重正则化防止模型过度拟合新类别

### 训练配置 | Training Configuration

| 参数 | 值 |
|------|-----|
| 数据集 | CIFAR-100 |
| 训练轮数 | 50 epochs |
| 骨干网络 | Vision Transformer |
| 优化器 | Adam / SGD |
| 学习率 | 动态调整 |

---

## 📊 结果分析 | Results Analysis

项目包含三种策略的训练结果：

1. **results_best/** — 最佳策略诊断结果
2. **results_comprehensive/** — 综合策略诊断结果
3. **results_progressive/** — 渐进式策略多轮训练结果（JSON 格式）

每个结果文件包含训练过程中的准确率、损失、混淆矩阵等诊断信息。

---

## 📚 参考文献 | References

- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- Rebuffi, S. A., et al. "iCaRL: Incremental Classifier and Representation Learning." CVPR 2017.
- Lopez-Paz, D., et al. "Gradient Episodic Memory for Continual Learning." NeurIPS 2017.

---

## 📄 License

MIT License — 自由使用、修改和分发。
