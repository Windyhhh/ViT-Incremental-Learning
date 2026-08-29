<div align="center">

# Vision Transformer 增量学习 | ViT-Incremental-Learning

### Vision Transformer incremental learning.

Class-incremental learning with ViT on CIFAR-100 — 50 epochs, progressive learning strategies.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

**ViT-Incremental-Learning** explores **class-incremental learning** with a **Vision Transformer (ViT)** on **CIFAR-100** — trained over 50 epochs with progressive learning strategies to mitigate catastrophic forgetting.

> [!NOTE]
> 中文项目：Vision Transformer 增量学习——CIFAR-100，50 epoch，渐进学习策略。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/ViT-Incremental-Learning.git
cd ViT-Incremental-Learning

pip install -r requirements.txt

# run incremental-learning training
python main.py
```

Results (diagnostic + progressive checkpoints) land in `results_*/`.

---

## Features

- **ViT incremental learning** — class-incremental on CIFAR-100.
- **Progressive strategies** — staged training runs.
- **Diagnostics** — result checkpoints and JSON logs.

---

## Project Structure

```
ViT-Incremental-Learning/
├── main.py                   # training entry
├── data/cifar-100-python/    # dataset
├── results_progressive/      # progressive results (JSON)
├── results_best/ / results_comprehensive/  # diagnostics
└── requirements.txt
```

---

## License

MIT — free to use, modify and distribute.
