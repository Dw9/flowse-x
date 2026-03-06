> ⚠️ **踩坑警告：采样率不匹配导致性能严重下降**
>
> VoiceBank_processed 数据集中的音频文件是 **48kHz** 采样率，但 FlowSE 模型是在 **16kHz** 频谱图上训练的（STFT 参数 `n_fft=510`, `hop_length=128` 均为 16kHz 设计）。
>
> `data_module.py` 的 `Specs` 数据集原本**没有做重采样**，直接在 48kHz 音频上做 STFT，导致频谱完全错误。`inference.py` 的 `evaluate_model()` 同样直接加载 48kHz 音频做推理，但 PESQ 却用 `sr=16000` 计算。
>
> **症状**：训练 PESQ 卡在 ~2.70（正确应 ≥3.09 baseline），评估 baseline checkpoint 只有 1.3（正确应为 3.15）。
>
> **修复**：在 `load_audio()` 中加入 `torchaudio.functional.resample(waveform, 48000, 16000)`，确保所有音频在进入 STFT 之前降采样到 16kHz。

# FlowSE: Flow matching based speech enhancement

* FlowSE: Flow Matching-based Speech Enhancement [1]



<p align="center">
  <a href="https://ieeexplore.ieee.org/document/10888274" target="_blank">
    <img src="https://seongqjini.com/wp-content/uploads/2025/02/Flowse_ani.gif" alt="FlowSE fig1" width="600"/>
  </a>
  
</p>
<p align="center">  
   <a href="https://youtu.be/sjYstc5ss-g?si=h7CSjjvYb3BwdT2f" target="_blank">
    <img src="https://img.youtube.com/vi/sjYstc5ss-g/0.jpg" width="600" alt="YouTube English Video"/>
  </a>
  <p align="center">
  <a href="https://youtu.be/sjYstc5ss-g?si=3yEjGvfJ4RdgKfuh">Presentation video [english]</a>, <a href="https://youtu.be/PI4qyd4YDJk?si=xhrrJ-MoRSewkQ36"> Presentation video [korean] </a>
  </a>
  </p>
</p>

This repository builds upon previous great works:
* [SGMSE] https://github.com/sp-uhh/sgmse  
* [SGMSE-CRP] https://github.com/sp-uhh/sgmse_crp
* [BBED]  https://github.com/sp-uhh/sgmse-bbed

## Installation
* Create a new virtual environment with Python 3.10 (we have not tested other Python versions, but they may work).
* Install the package dependencies via ```pip install -r requirements.txt```.


## Training
Training is done by executing train.py. A minimal running example with default settings (as in our paper [1]) can be run with

```bash
python train.py --base_dir <your_dataset_dir>
```
where `your_dataset_dir` should be a containing subdirectories `train/` and `valid/` (optionally `test/` as well). 

Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.

To get the training set WSJ0-CHIME3, we refer to https://github.com/sp-uhh/sgmse and execute create_wsj0_chime3.py.

To see all available training options, run python train.py --help. 

## Evaluation
  To evaluate on a test set, run


  ```bash
  python evaluate.py --test_dir <your_test_dataset_dir> --folder_destination <your_enh_result_save_dir> --ckpt <path_to_model_checkpoint> --N <num_of_time_steps>
  ```

`your_test_dataset_dir` should contain a subfolder `test` which contains subdirectories `clean` and `noisy`. `clean` and `noisy` should contain .wav files.

Continuous Normalizing Flow (CNF) is a method transforming a simple distribution $p(x)$ to a complex distribution $q(x)$.  

CNF is described by Ordinary Differential Equations (ODEs):  

$$
\frac{d \phi_t(x_0)}{dt} = v(t,\phi_t(x_0)), \quad \phi_0(x_0)=x_0, \quad x_0\sim p(\cdot)
$$

In the above ODE, a function $\phi_t$ called flow is desired such that the stochastic process $x_t=\phi_t(x_0)$ has a marginal distribution $p_t(\cdot)$ such that $p_1(\cdot ) = q(\cdot)$.  

In the above equation, although the condition that $\phi_0(x_0)$ follows $p$ is imposed (initial value problem), by chain rule replacing $t$ with $1-t$, CNF can be described as:  

$$
\frac{d\phi_t(x_1)}{dt} = v_t(t,\phi_t(x_1)), \quad \phi_1(x_1)=x_1, \quad x_1 \sim p(\cdot)
$$

It means that it does not matter that the simple distribution is located at which time point.  

# Part 2: DriftingSE — 基于 Drifting Models 的单步语音增强

---

## 1. 概述

DriftingSE 将 **Drifting Models**（Deng et al., 2026, arXiv:2602.04770, Kaiming He 团队）的思想应用于语音增强任务。原始 Drifting 论文在 ImageNet 256×256 上以单步（1-NFE）生成达到 FID 1.54，其核心创新在于训练过程中让生成分布自然演化，使模型天然支持单步推理。

**与 FlowSE 的关键区别：**

| | FlowSE (Part 1) | DriftingSE (Part 2) |
|---|---|---|
| 推理方式 | 多步 ODE 求解（默认 N=5） | **单次前向传播**（1-NFE） |
| 训练目标 | 学习速度场 $v_\theta$，沿 ODE 轨迹积分 | 学习直接映射 noisy → clean |
| Loss | 速度场 MSE | 混合 loss（drifting + MSE） |
| 时间条件 | ODE 时间步 $t$ | 输入能量自适应条件 |
| 推理速度 | ~5x backbone 前向 | **1x backbone 前向** |

DriftingSE 复用 FlowSE 的 NCSNpp backbone 和 STFT 数据管线，确保公平对比。

 唯一的语义差异

  ┌──────────┬─────────────────────────┬─────────────────────────────────┐
  │          │         FlowSE          │           DriftingSE            │
  ├──────────┼─────────────────────────┼─────────────────────────────────┤
  │ 输入 t   │ 随机采样 t ∈ [0,1]（ODE │ sigmoid(log_energy)（输入能量） │
  │          │  时间步）               │                                 │
  ├──────────┼─────────────────────────┼─────────────────────────────────┤
  │ 输出含义 │ 速度场 v(x_t, y,        │ 直接估计 x̂ = y + backbone(y, 0) │
  │          │ t)（ODE 梯度方向）      │                                 │
  ├──────────┼─────────────────────────┼─────────────────────────────────┤
  │ 任务     │ 预测如何从 x_t 移向 x_0 │ 直接从 y 预测 x_0      
---

## 2. 方法

### 2.1 Drifting Field 理论（Algorithm 2）

Drifting 的核心思想：定义一个向量场 $V_{p,q}(x)$ 来衡量生成分布 $q$ 与目标分布 $p$ 之间的差异，并驱动 $q$ 向 $p$ 演化。

$$V_{p,q}(x) = V_p^+(x) - V_q^-(x)$$

- $V_p^+$（正样本吸引力）：将生成样本拉向真实数据分布
- $V_q^-$（负样本排斥力）：将生成样本推离当前生成分布
- **反对称性保证**：当 $q = p$ 时 $V = 0$（平衡态），loss 自然归零

具体计算过程（`compute_drifting_field()`）：

1. 计算生成样本 $x$ 与正样本 $y_{pos}$（clean）、负样本 $y_{neg}$（generated）之间的 L2 距离
2. 应用 softmax kernel：$\text{logit} = -\text{dist} / \tau$，对正负样本联合归一化
3. 双向归一化：$A = \sqrt{A_{row} \cdot A_{col}}$（行归一化 × 列归一化的几何平均）
4. 计算加权 drift 向量：$V = W_{pos} \cdot y_{pos} - W_{neg} \cdot y_{neg}$
5. 负样本采用 self-negative 策略（$y_{neg} = x$），对角线 mask 避免自排斥

**Drifting Loss**：

$$L_{drift} = \| \phi(x) - \text{sg}(\phi(x) + V) \|^2$$

其中 $\phi$ 为多尺度特征提取器，$\text{sg}$ 为 stop-gradient。梯度只通过 $\phi(x)$ 流向生成器，目标 $\phi(x) + V$ 被冻结。

### 2.2 混合 Loss

$$L_{total} = \lambda_{drift} \cdot L_{drift} + \lambda_{recon} \cdot L_{recon}$$

- **$L_{drift}$**（分布匹配）：多尺度 drifting loss，驱动输出分布整体对齐到 clean 分布
- **$L_{recon}$**（逐样本保真）：$\text{mean}(|x_{hat} - x_{clean}|^2)$，确保每条语音的细节还原

两者互补：drift 提供全局分布正则化，MSE 保证逐样本精度。

### 2.3 残差学习

$$\hat{x} = y + \text{backbone}(y, \epsilon)$$

backbone 初始化时输出接近零，因此 $\hat{x} \approx y$，这是语音增强的合理起点（带噪语音本身包含大部分干净信号）。模型只需学习预测残差（噪声），而非从头生成完整频谱。

### 2.4 能量自适应条件（Energy-Adaptive Conditioning）

NCSNpp 原本使用 ODE 时间步 $t$ 通过 Fourier embedding 和 FiLM 层调制 backbone。DriftingSE 中没有 ODE 时间步，取而代之的是：

$$t = \sigma\left(\log\left(\text{mean}(|y|)\right)\right)$$

其中 $\sigma$ 为 sigmoid 函数，$|y|$ 为输入频谱的幅度。

**效果**：不同噪声水平的输入产生不同的 $t$ 值，NCSNpp 的 16 个 ResNet block 通过 FiLM 层得到不同的调制，从而自适应地处理不同噪声条件。

### 2.5 多尺度特征提取

DriftingSE 的 drifting loss 在多尺度特征空间上计算，包含两大类共 14 个尺度：

**原始频谱特征（7 个尺度）**，由 `extract_multiscale_features()` 提取：

| 尺度 | 描述 | 维度 |
|------|------|------|
| (a) 全局统计 | mean/std over F×T | $[B, 2C]$ |
| (b) 频率维均值 | mean over T per freq | $[B, C \times F]$ |
| (c) 频率维方差 | std over T per freq | $[B, C \times F]$ |
| (d) 时间维均值 | mean over F per frame | $[B, C \times T]$ |
| (e) 时间维方差 | std over F per frame | $[B, C \times T]$ |
| (f) 4×4 patch 均值 | local patch statistics | $[B, C \times (F/4) \times (T/4)]$ |
| (g) 能量统计 | mean of squared values | $[B, C]$ |

**Mel 频谱特征（7 个尺度）**，由 `extract_mel_features()` 提取：

| 尺度 | 描述 | 维度 |
|------|------|------|
| (a) Band 均值 | log-mel per-band mean | $[B, n_{mels}]$ |
| (b) Band 方差 | log-mel per-band std | $[B, n_{mels}]$ |
| (c) Frame 能量 | log-mel per-frame mean | $[B, T]$ |
| (d) Frame 方差 | log-mel per-frame std | $[B, T]$ |
| (e) Mel patch 统计 | 4×4 mel patch mean | $[B, (n_{mels}/4) \times (T/4)]$ |
| (f) Delta 均值 | temporal change per band | $[B, n_{mels}]$ |
| (g) Delta 方差 | temporal variability | $[B, n_{mels}]$ |

Mel 特征路径：$|\text{spec}| \to \text{mel filterbank} \to \log\text{-mel} \to$ 各种统计量。全程在 STFT 域完成，无需 ISTFT。

每个尺度的特征先做 Section A.6 归一化（使平均距离 $\approx \sqrt{D}$），再分别计算 drifting loss，最后取所有尺度的平均。

### 2.6 FeatureBank：正样本增广

`FeatureBank` 是一个环形缓冲区（ring buffer），在训练中存储历史 batch 的 clean 特征。

- **容量**：`max_size=256`（默认），每个 DDP 进程独立维护
- **存储**：仅存储正样本（clean 特征），因其不依赖模型，不会过时
- **用途**：增广 drifting loss 中的正样本集合，使分布估计更准确
- **内存**：约 12MB，存储在 CPU 上以节省 GPU 显存

---

## 3. 训练指标说明

### 核心 Loss（每个均有 step 和 epoch 两个粒度）

| 指标 | 含义 | 正常范围 |
|------|------|----------|
| `drift_loss` | $\|\|V\|\|^2$ — drifting 向量的平方模长。生成分布与 clean 分布越远越大，完全对齐时为 0 | 训练初期 ~10+，逐渐下降 |
| `recon_loss` | $\text{mean}(\|x_{hat} - x_{clean}\|^2)$ — 逐样本 MSE 重建损失 | ~0.002 量级 |
| `train_loss` | $\lambda_{drift} \times \text{drift} + \lambda_{recon} \times \text{recon}$ — 加权总 loss | 取决于各权重 |
| `valid_loss` | 同 train_loss，但使用 EMA 权重在验证集上计算 | 应略低于 train_loss |

### 监控指标

| 指标 | 含义 | 用途 |
|------|------|------|
| `drift_recon_ratio` | $\text{drift\_loss} / \text{recon\_loss}$ | 观察哪个 loss 主导训练。比值过大说明 drift 主导，可能需要降低 drift_weight |
| `drift_v_magnitude` | $\|\|V\|\|$（非平方），drifting 向量的实际大小 | 跟踪收敛状态。训练过程中应逐渐下降，收敛时趋近于 0 |
| `pesq` | 宽带 PESQ（16kHz），语音质量的客观指标 | 核心评价指标，越高越好（1.0~4.5） |
| `si_sdr` | Scale-Invariant Signal-to-Distortion Ratio (dB) | 信号失真度，越高越好 |
| `estoi` | Extended Short-Time Objective Intelligibility | 语音可懂度，越高越好（0~1） |

---

## 4. 关键超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `--drift_weight` | **0.008**（推荐） | drifting loss 的权重。通过梯度范数分析确定：drift/recon 梯度比约为 260:1，因此需要很小的 drift_weight 来平衡梯度。dw=0.004 使梯度均衡，dw=0.008（drift 梯度约 2 倍于 recon）效果最佳 |
| `--recon_weight` | 1.0 | MSE 重建 loss 的权重 |
| `--loss_type` | `hybrid` | 可选 `hybrid`（推荐）/ `drifting`（纯 drifting）/ `mse`（纯 MSE baseline） |
| `--accumulate_grad_batches` | 4 | 梯度累积步数。有效 batch = batch_size × GPU 数 × 4 |
| `--bank_size` | 256 | FeatureBank 容量。设为 0 禁用 |
| `--temperatures` | `0.02,0.05,0.2` | softmax kernel 的温度值。每个温度独立计算 drift 向量后累加。低温关注近邻，高温关注全局结构 |
| `--energy_conditioning` | 1 | 是否使用能量自适应条件。1 = 启用，0 = 固定 t=1.0 |
| `--use_residual` | 1 | 是否使用残差学习。1 = $\hat{x} = y + \text{backbone}$，0 = $\hat{x} = \text{backbone}$ |
| `--use_mel_features` | 1 | 是否在 drifting loss 中使用 mel 特征（14 尺度 vs 仅 7 尺度） |
| `--n_mels` | 80 | mel 滤波器组的频带数 |
| `--ema_decay` | 0.999 | EMA 权重的指数衰减系数 |
| `--lr` | 1e-4 | 学习率（Adam 优化器） |

---

## 5. 实验结果

在 VoiceBank 数据集上的实验（约 115 epochs，验证集评估）：

| 配置 | PESQ (peak) | SI-SDR | recon_loss (val) | 说明 |
|------|:-----------:|:------:|:----------------:|------|
| A2: MSE baseline | 2.10 | 15.97→9.1 | 0.0026 | ep132 后训练崩溃 |
| B2: hybrid dw=0.004 | 2.09 | 14.65 ↑ | 0.0023 | 梯度均衡点 |
| **B3: hybrid dw=0.008** | **2.17** | **14.77** ↑ | **0.0019** | drift 梯度约 2 倍于 recon |
| C1: hybrid dw=0.01 | 2.10 / 2.29 (test) | 15.75 (test) | — | 完整 test set 评估 |

### 关键发现

1. **Drifting loss 确实有效**：B3 (hybrid) 在 PESQ 上优于 A2 (pure MSE)，同时 recon_loss 更低
2. **正则化效果**：drift loss 防止了纯 MSE 训练的崩溃（A2 在 ep132 后 SI-SDR 从 15.97 骤降至 9.1）
3. **梯度平衡是关键**：原始 drift/recon 梯度比为 ~260:1，需要小 drift_weight 来平衡。最佳点在 drift 梯度略大于 recon 梯度时（dw=0.008）
4. **单步推理**：所有 DriftingSE 实验均为 1-NFE（单次前向传播），推理速度约为 FlowSE（N=5）的 5 倍

---

## 6. 训练与评估命令

### 训练

```bash
# 混合 loss（推荐配置）：drift_weight=0.008
python train_drifting.py \
    --base_dir /path/to/VoiceBank_processed \
    --loss_type hybrid \
    --drift_weight 0.008 \
    --no_wandb

# 纯 MSE baseline（对照实验）
python train_drifting.py \
    --base_dir /path/to/VoiceBank_processed \
    --loss_type mse \
    --no_wandb

# 纯 drifting loss（实验性质）
python train_drifting.py \
    --base_dir /path/to/VoiceBank_processed \
    --loss_type drifting \
    --drift_weight 1.0 \
    --no_wandb

# 从预训练权重初始化（例如 FlowSE checkpoint）
python train_drifting.py \
    --base_dir /path/to/VoiceBank_processed \
    --loss_type hybrid \
    --drift_weight 0.008 \
    --ckpt /path/to/pretrained.ckpt \
    --no_wandb

# 使用 W&B 日志（去掉 --no_wandb，项目名为 DRIFTING-SE）
python train_drifting.py \
    --base_dir /path/to/VoiceBank_processed \
    --loss_type hybrid \
    --drift_weight 0.008
```

### 评估

```bash
# 在测试集上评估（单步推理，1-NFE）
python eval_drifting.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --data_dir /path/to/VoiceBank_processed \
    --split test

# 评估并保存增强后的音频
python eval_drifting.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --data_dir /path/to/VoiceBank_processed \
    --split test \
    --output_dir /path/to/output
```

### Checkpoint 保存策略

训练过程自动保存三类 checkpoint：
- `{epoch}_last.ckpt`：最新一轮
- `{epoch}_{pesq:.2f}.ckpt`：PESQ 最高的 top-20
- `{epoch}_{si_sdr:.2f}.ckpt`：SI-SDR 最高的 top-20

保存路径：`logs/drifting_<dataset>_<loss_type>_dw<weight>_<KST_timestamp>/`

---

## 7. 代码结构

```
flowmse/
├── drifting.py            # 核心 drifting field 计算
│                          #   - FeatureBank: 正样本特征环形缓冲区
│                          #   - compute_drifting_field(): Algorithm 2 实现
│                          #   - normalize_features(): Section A.6 特征归一化
│                          #   - drifting_loss_single_scale(): 单尺度 drifting loss
│                          #   - create_mel_filterbank(): mel 滤波器组
│                          #   - extract_mel_features(): mel 频谱特征（7 尺度）
│                          #   - extract_multiscale_features(): 原始+mel 特征（14 尺度）
│                          #   - compute_drifting_loss(): 多尺度 drifting loss 入口
│
├── drifting_model.py      # DriftingSEModel (PyTorch Lightning module)
│                          #   - 单步前向传播：x_hat = y + backbone(y, ε)
│                          #   - 能量自适应条件：t = sigmoid(log(mean(|y|)))
│                          #   - 混合 loss 计算与日志记录
│                          #   - 手动 EMA 权重管理
│                          #   - 验证时 PESQ/SI-SDR/ESTOI 评估
│
├── backbones/
│   └── ncsnpp.py          # NCSNpp U-Net backbone（与 FlowSE 共享）
│                          #   FiLM 条件由时间步改为输入能量驱动
│
├── data_module.py         # 数据加载（STFT、重采样等，与 FlowSE 共享）
│
train_drifting.py          # 训练脚本
│                          #   - 支持 hybrid / drifting / mse 三种 loss 模式
│                          #   - 默认梯度累积 4 步
│                          #   - TensorBoard 或 W&B 日志
│                          #   - 支持从预训练 checkpoint 初始化
│
eval_drifting.py           # 评估脚本
                           #   - 单步推理（1-NFE），无需 ODE 求解
                           #   - 输出 PESQ / SI-SDR / ESTOI
                           #   - 可选保存增强后音频
```

**与 FlowSE 的代码关系**：DriftingSE 的代码完全独立于 FlowSE 的训练/推理流程（`model.py`, `train.py`, `evaluate.py`）。两者共享 backbone（`ncsnpp.py`）、数据管线（`data_module.py`）和工具函数（`util/`），但互不影响。
