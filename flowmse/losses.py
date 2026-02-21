"""
Loss functions for flow matching-based speech enhancement.

Based on: "Investigating Training Objectives for Flow Matching-based Speech Enhancement"
https://arxiv.org/pdf/2512.10382v1

This module implements:
- SI-SDR Loss: Scale-invariant signal-to-distortion ratio loss
"""

import torch


def si_sdr_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.

    Args:
        pred: Predicted signal, shape (batch, time) or (batch, channel, time)
        target: Target signal, shape (batch, time) or (batch, channel, time)
        eps: Small constant for numerical stability

    Returns:
        SI-SDR loss (negative SI-SDR in dB, so minimizing this maximizes SI-SDR)

    Formula from paper:
        L_SI-SDR(x̂1, x1) = -10 * log10(||ω*x1||² / ||x̂1 - ω*x1||²)
        where ω = x̂1^T @ x1 / (x1^T @ x1)
    """
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    dot_target_pred = (target * pred).sum(dim=-1)
    dot_target_target = (target * target).sum(dim=-1)

    omega = dot_target_pred / (dot_target_target + eps)
    scaled_target = omega.unsqueeze(-1) * target
    error = pred - scaled_target

    numerator = (scaled_target**2).sum(dim=-1)
    denominator = (error**2).sum(dim=-1)

    si_sdr = 10 * torch.log10(numerator / (denominator + eps) + eps)

    return -si_sdr.mean()


def estimate_x0_from_velocity(
    xt: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Estimate clean speech x0 from velocity prediction.

    This codebase's convention (odes.py):
        µ_t = (1-t)*x0 + t*y,  σ_t = t*σ_max  (σ_min=0)
        x_t = (1-t)*x0 + t*y + t*σ_max*z
        condVF = (y - x0) + σ_max*z

    Derivation:
        x_t - t*v_θ = (1-t)*x0 + t*y + t*σ_max*z - t*(y - x0 + σ_max*z) = x0

    NOTE: The paper (2512.10382) uses the OPPOSITE time convention
        (t=0 noisy, t=1 clean), where x̂₁ = x_t + (1-t)*v_θ.
        This codebase uses t=0 clean, t=1 noisy, so the formula differs.

    Args:
        xt: Interpolated spectrogram at time t
        velocity: Predicted velocity field v_θ(xt, y, t)
        t: Time values, shape (batch,)

    Returns:
        Estimated clean spectrogram x0
    """
    t_factor = t.view(-1, 1, 1, 1)  # (batch, 1, 1, 1)
    x0_hat = xt - t_factor * velocity
    return x0_hat
