# Quantum Annealing for Mixed-Precision Neural Network Quantization

基于量子退火优化的混合精度神经网络量化方案

## 项目简介

将 DistilBERT 的混合精度分配问题建模为 QUBO（二次无约束二进制优化），通过相干光量子退火求解。核心创新：**动态加权层间平滑约束**——浅层严控精度跳变，深层逐步放宽，引导优化器自动生成硬件友好的连续精度块。

## 技术栈

- Python 3.10
- PyTorch + HuggingFace Transformers（加载 DistilBERT）
- Kaiwu SDK（相干光量子真机接口）
- NumPy + Matplotlib（实验与可视化）

## 目录结构

```
quantum_project/
├── qubo_model.py          # QUBO 矩阵构造（核心）
├── quantize_solver.py     # SA 求解 + 量子接口 + 解码
├── model_inspect.py       # DistilBERT 层结构分析
├── eval_compare.py        # 七策略对比实验（主实验脚本）
├── tune_smooth.py         # λ_s 甜点扫描
├── quantum_compare.py     # SA vs 量子对比
├── fair_compare.py        # 公平对比（超低耦合度编码）
├── _draw_diagrams.py      # 图表绘制
├── _plot_weights.py       # 权重曲线绘制
├── config.example.py      # 配置文件模板（请复制为 config.py 并填入自己的密钥）
├── output/                # 实验输出图表
└── README.md
```

## 快速开始

1. 安装依赖：
```bash
pip install torch transformers numpy matplotlib kaiwu
```

2. 配置密钥：
```bash
cp config.example.py config.py
# 编辑 config.py 填入 ACCESS_KEY / SECRET_KEY
```

3. 运行主实验：
```bash
python eval_compare.py
```

4. 纯 SA 模式（不连量子真机）：
直接运行 `eval_compare.py` 即可，SA 部分不依赖 Kaiwu API。

## 实验规模

- 模型：DistilBERT（66M 参数，13 个决策组）
- 精度档位：FP16 / INT8 / INT6 / INT4 / INT2
- 搜索空间：5¹³ ≈ 12 亿种组合
- QUBO 矩阵：65×65
- 共 10 篇参考文献

## 论文

见 `paper.pdf`（LaTeX 源码在 `E:\forlatex\template\paper.tex`）

## 许可

MIT License
