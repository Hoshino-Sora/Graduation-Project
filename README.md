# Deep-Learning-Epileptic-Seizure-Classification

> **基于深度学习的长程多通道癫痫发作分类方法研究** > *A Robust Deep Learning Framework for Long-term EEG Epileptic Seizure Detection*

本项目致力于解决真实临床环境下长程脑电信号（EEG）自动化监测的核心痛点：跨患者泛化困难与极度不平衡数据分布下的高频误报。通过构建稳健的时空特征网络与事件级后处理机制，实现高灵敏度、低误报率的临床级癫痫检测。

---

## 🚀 项目亮点 (Key Features)

* **工业级数据管道 (Data Pipeline)**：实现了针对真实临床长程 EDF 格式数据的自动化清洗流水线。彻底解决了多通道设备在监护中途变更（Channels Changed）导致的维度崩溃问题。
* **标准化通道对齐 (Channel Alignment)**：严格提取国际 10-20 系统的 18 个核心双极导联通道，物理阻断跨个体通道差异，为模型的强泛化能力奠定基础。
* **内存防爆栈策略 (OOM Prevention)**：采用滑动切窗与分批持久化落盘（`.npy`）策略，完美规避处理数百 GB 级 CHB-MIT 数据时的内存溢出风险。

---

## 📁 核心架构 (Project Structure)

项目采用高度模块化的“前后台分离”架构：

* `config.py`: **全局配置中枢**。统一管理采样率、通道白名单、滑动窗口长度（默认2s）等超参数，杜绝硬编码。
* `data_loader.py`: **解析引擎**。包含对 Bonn 数据集（TXT）和 CHB-MIT 数据集（EDF/Summary 文本正则化提取）的底层解析函数。
* `preprocess.py`: **预处理车间**。负责 Z-score 标准化与正负样本标签（ictal / non-ictal）的严格时序对齐。
* `build_chbmit_dataset.py`: **全库批处理调度脚本**。基于配置参数动态生成数据集切片并落盘至本地硬盘。
* `models.py`: *(In Progress)* 存放双分支特征提取与时空深度网络架构（如 TCN-BiLSTM）。
* `train.py` & `evaluate.py`: *(In Progress)* 闭环训练与事件级指标（如 FD/h）评估。

---

## 📊 数据集支持 (Datasets)

本项目依托两大国际公开基准数据集进行递进式验证：

1.  **Bonn 数据集**：切分规整的单通道无背景噪声片段，用于基础算法与特征工程的初期可行性验证。
2.  **CHB-MIT 数据库**：源自波士顿儿童医院的长程多通道连续脑电记录，真实还原了极度不平衡（非发作期占比极高）的临床痛点，作为本项目的核心性能评估基准。

---

## 🛠️ 环境依赖 (Environment Setup)

请确保你的 Python 环境安装了以下核心依赖：

```bash
pip install numpy pandas scikit-learn mne pywt
```
*(注：深度学习框架 PyTorch 依赖将在进入模型开发阶段后引入。`pywt` 为后续小波变换特征提取库。)*

---

## 🚦 快速开始 (Quick Start)

**步骤 1：准备数据**
将下载好的 CHB-MIT 数据集解压至项目根目录的 `datasets/chbmit/` 下。

**步骤 2：全量预处理与特征切片**
配置好 `config.py` 中的参数后，在终端执行批处理脚本：
```bash
python build_chbmit_dataset.py
```
预处理后的净数据（`.npy` 矩阵）将根据你的窗口参数（如 `win2s_ov0s`）动态生成专属文件夹。

---

## 📅 进度追踪 (Progress & Roadmap)

- [x] **Week 1-2**: 环境搭建，完成 Bonn 与 CHB-MIT 多源数据集的解析与切片持久化流水线。
- [ ] **Week 3**: 基线模型构建，完成基于离散小波变换 (DWT) 的传统集成学习基线。
- [ ] **Week 4-5**: 构建端到端深度学习检测基线（TCN-BiLSTM），引入后处理框架。
- [ ] **Week 6-7**: 改进算法融合（MixUp 增强与双分支网络），深入消融实验对比。
