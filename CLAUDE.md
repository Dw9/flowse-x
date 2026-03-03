# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlowSE is a **Flow Matching-based Speech Enhancement** system (ICASSP 2025). It uses Continuous Normalizing Flows (CNFs) via ODEs to transform noisy speech spectrograms into clean ones. The model learns a velocity field defining the ODE trajectory from noisy to clean speech. Built on top of the SGMSE/SGMSE+ codebase from sp-uhh.

## Commands

### Environment Setup
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
python train.py --base_dir <dataset_dir> [--backbone ncsnpp|dcunet] [--no_wandb] [--ckpt <path>]
```
Dataset dir must contain `train/`, `valid/` (optionally `test/`) subdirs, each with `clean/` and `noisy/` subdirs of matching `.wav` files.

### Evaluation
```bash
# Detailed evaluation with per-file metrics
python evaluate.py --test_dir <dataset_dir> --folder_destination <output_dir> --ckpt <checkpoint_path> --N 5

# Simplified full test-set evaluation (handles 48kHz→16kHz resampling)
python eval_full.py --checkpoint <checkpoint_path> --data_dir <dataset_dir> --split test
```

### No test suite
There are no automated tests. Validation is done via evaluation scripts against held-out test sets.

## Architecture

### Core Flow (Training → Inference)
1. **Data**: `data_module.py` loads clean/noisy `.wav` pairs, resamples to 16kHz, computes STFT spectrograms (`n_fft=510`, `hop_length=128`), applies `abs^0.5 * exp(jθ) * 0.15` compression
2. **Training**: `VFModel` (Lightning module in `model.py`) samples random timestep `t`, computes marginal state `x_t` via the flow matching ODE, predicts velocity field with backbone DNN, optimizes MSE/MAE loss
3. **Inference**: Euler ODE solver steps from `t=1` (noisy) → `t≈0` (clean) with N steps (default 5)

### Critical Time Convention
**Opposite of Lipman et al.**: `t=0` = clean speech, `t=1` = noisy speech. Inference runs backwards from 1→0. Clean estimate: `x0_hat = x_t - t * v_θ(x_t, y, t)`.

### Key Components
- **`flowmse/model.py`** — `VFModel`: PyTorch Lightning module. Central hub for training, validation, inference. Manual EMA implementation for backbone weights. Swaps to EMA weights during eval.
- **`flowmse/odes.py`** — `FLOWMATCHING` ODE: defines probability path (`mu_t = (1-t)*x0 + t*y`, `sigma_t = (1-t)*sigma_min + t*sigma_max`)
- **`flowmse/backbones/ncsnpp.py`** — Default backbone. NCSN++ U-Net operating on 4-channel real-valued input (real/imag of x and y). Fourier time embedding, BigGAN ResNet blocks, attention at resolution 16.
- **`flowmse/backbones/dcunet.py`** — Alternative complex-valued U-Net backbone (DCUNet-10/16/20 variants)
- **`flowmse/sampling/`** — White-box Euler solver and black-box scipy RK45 wrapper

### Registry Pattern
Backbones, ODEs, and solvers are registered by name via `Registry` class (`flowmse/util/registry.py`). Registered names: backbones=`"ncsnpp"`,`"dcunet"`; ODEs=`"flowmatching"`; solvers=`"euler"`.

### Logging & Checkpoints
- Default: Weights & Biases (project "FLOWSE"); `--no_wandb` switches to TensorBoard
- Checkpoints saved in `logs/dataset_<name>_<KST_timestamp>/` — by best PESQ, best SI-SDR, and last epoch
- Metrics: PESQ (wideband 16kHz), ESTOI, SI-SDR

## Pitfalls

- **Sample rate**: Model expects 16kHz. VoiceBank dataset is 48kHz — `load_audio()` handles resampling but be aware when adding new data sources.
- **STFT params are hardcoded** for 16kHz in `data_module.py` and `util/inference.py`.
- **EMA weights**: The model uses manual EMA (not torch_ema). During validation/test, `_swap_ema()` is called to use EMA weights. Legacy migration code exists for old checkpoints.
