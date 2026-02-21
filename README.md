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

# Part 2: Investigating Training Objectives for Flow Matching-based Speech Enhancement

* Investigating Training Objectives for Flow Matching-based Speech Enhancement [2]

Based on the FlowSE framework above, we integrate perceptual (PESQ) and signal-based (SI-SDR) auxiliary losses proposed in [2] to further enhance convergence efficiency and speech quality.

## Method

The core idea is to add auxiliary losses that directly optimize perceptual metrics during flow matching training:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CFM}} + \alpha_p \cdot \mathcal{L}_{\text{PESQ}} + \alpha_s \cdot \mathcal{L}_{\text{SI-SDR}}
$$

To compute the auxiliary losses, we need to estimate the clean speech $\hat{x}_0$ from the model's velocity prediction $v_\theta(x_t, y, t)$ at each training step, then convert it to the time domain via iSTFT and evaluate PESQ / SI-SDR against the ground truth.

### Recommended hyperparameters from [2]

| Training Objective | $\alpha_p$ (PESQ) | $\alpha_s$ (SI-SDR) |
|---|---|---|
| Velocity prediction ($\mathcal{L}_{\text{CFM-v}}$) | 5e-2 | 5e-3 |
| $x_1$ prediction ($\mathcal{L}_{\text{CFM-}x_1}$) | 1e-3 | 1e-4 |
| Preconditioned $x_1$ ($\mathcal{L}_{\text{CFM-}x_1\text{-EDM}}$) | 1e-6 | 1e-7 |

This codebase currently uses velocity prediction, so the defaults are $\alpha_p = 5 \times 10^{-2}$, $\alpha_s = 5 \times 10^{-3}$.

## ⚠️ Critical: Time Convention Mismatch

**This codebase and [2] use opposite time conventions.** Directly copying formulas from [2] without adaptation will produce incorrect results.

### The two conventions

| | This codebase (from SGMSE/Score SDE tradition) | Paper [2] (from Lipman et al. FM tradition) |
|---|---|---|
| $t = 0$ | Clean speech $x_0$ | Noisy speech $y$ |
| $t = 1$ | Noisy speech $y$ | Clean speech $x_1$ |
| $\mu_t$ | $(1-t) \cdot x_0 + t \cdot y$ | $t \cdot x_1 + (1-t) \cdot y$ |
| $\sigma_t$ | $t \cdot \sigma_{\max}$ | $(1-t) \cdot \sigma_{\max}$ |
| Inference | $t: 1 \to 0$ (reverse) | $t: 0 \to 1$ (forward) |

Mathematically these are **identical** — just relabel $t \leftrightarrow (1-t)$. But formulas involving $t$ must be adapted.

### Estimating clean speech $\hat{x}_0$ from velocity

In **[2]'s convention** (t=0 noisy, t=1 clean):
$$
\hat{x}_1 = x_t + (1-t) \cdot v_\theta(x_t, y, t)
$$

In **this codebase's convention** (t=0 clean, t=1 noisy):
$$
\hat{x}_0 = x_t - t \cdot v_\theta(x_t, y, t)
$$

### Derivation (this codebase)

Given (with $\sigma_{\min} = 0$):

$$
x_t = (1-t) \cdot x_0 + t \cdot y + t \cdot \sigma_{\max} \cdot z
$$
$$
\text{condVF} = (y - x_0) + \sigma_{\max} \cdot z
$$

Verify $x_t - t \cdot \text{condVF}$:

$$
\begin{aligned}
&\quad [(1-t) x_0 + t \cdot y + t \sigma_{\max} z] - t \cdot [(y - x_0) + \sigma_{\max} z] \\
&= (1-t) x_0 + t \cdot y + t \sigma_{\max} z - t \cdot y + t \cdot x_0 - t \sigma_{\max} z \\
&= (1-t) x_0 + t \cdot x_0 \\
&= x_0 \quad \checkmark
\end{aligned}
$$

### What goes wrong with the naive formula

| Formula | Result | Correct? |
|---|---|---|
| $x_t + (1-t) \cdot v_\theta$ (paper [2] formula, wrong here) | $y + \sigma_{\max} z$ | ❌ noisy + extra noise |
| $y - v_\theta$ (simplified, ignoring $\sigma_{\max}$) | $x_0 - \sigma_{\max} z$ | ❌ clean + residual noise |
| $x_t - t \cdot v_\theta$ **(correct for this codebase)** | $x_0$ | ✅ exact recovery |

## Training with auxiliary losses

```bash
python train.py --base_dir <your_dataset_dir> \
    --use_pesq_loss --use_si_sdr_loss \
    --alpha_pesq 5e-2 --alpha_si_sdr 5e-3
```

You can also enable only one of the two:

```bash
# PESQ loss only
python train.py --base_dir <your_dataset_dir> --use_pesq_loss --alpha_pesq 5e-2

# SI-SDR loss only
python train.py --base_dir <your_dataset_dir> --use_si_sdr_loss --alpha_si_sdr 5e-3
```

### Additional dependency

PESQ loss requires the differentiable [torch-pesq](https://github.com/audiolabs/torch-pesq) package:

```bash
pip install torch-pesq
```

## Implementation

The auxiliary loss implementation lives in:

- `flowmse/losses.py` — SI-SDR loss function and `estimate_x0_from_velocity`
- `flowmse/model.py` — Integration into the training loop (`_step` method)

## References

- [1] S. Lee, S. Cheong, S. Han, J. W. Shin, "FlowSE: Flow Matching-based Speech Enhancement," *ICASSP 2025*. [arXiv:2508.06840](https://arxiv.org/abs/2508.06840)
- [2] L. Yang, Z. Ge, G. Zhang, J. Zhang, Z. Wu, "Investigating Training Objectives for Flow Matching-based Speech Enhancement," *arXiv 2025*. [arXiv:2512.10382](https://arxiv.org/abs/2512.10382)

