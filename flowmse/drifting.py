"""
Drifting field computation for Drifting Models applied to Speech Enhancement.

Based on: "Generative Modeling via Drifting" (Deng et al., 2026)
Algorithm 2: Computing the drifting field V.

Core idea: V_{p,q}(x) = V_p^+(x) - V_q^-(x)
  - V_p^+: attracts generated samples toward data distribution
  - V_q^-: repels generated samples from their own distribution
  - Anti-symmetry guarantees V=0 when q=p (equilibrium)
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_drifting_field(x, y_pos, y_neg, temperature, self_neg=True):
    """
    Compute drifting field V following Algorithm 2 of the Drifting paper.

    Args:
        x: [N, D] features of generated samples
        y_pos: [N_pos, D] features of positive (real data) samples
        y_neg: [N_neg, D] features of negative (generated) samples
        temperature: float, kernel temperature (pre-scaled by sqrt(D))
        self_neg: bool, if True, y_neg is x itself so mask diagonal

    Returns:
        V: [N, D] drifting field vectors
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]

    # Compute pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos, p=2)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg, p=2)  # [N, N_neg]

    # Mask self-distances if y_neg is x (avoid trivial self-repulsion)
    if self_neg and N == N_neg:
        dist_neg = dist_neg + torch.eye(N, device=x.device, dtype=x.dtype) * 1e6

    # Compute logits (negative distance / temperature)
    logit_pos = -dist_pos / temperature  # [N, N_pos]
    logit_neg = -dist_neg / temperature  # [N, N_neg]

    # Concatenate for joint normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]

    # Normalize along both dimensions (softmax)
    A_row = F.softmax(logit, dim=1)   # normalize over y (pos+neg)
    A_col = F.softmax(logit, dim=0)   # normalize over x
    A = torch.sqrt(A_row * A_col + 1e-12)

    # Split back to positive and negative parts
    A_pos = A[:, :N_pos]   # [N, N_pos]
    A_neg = A[:, N_pos:]   # [N, N_neg]

    # Compute weights (Eq. 11 decomposed)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # [N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # [N, N_neg]

    # Compute drift vectors
    drift_pos = W_pos @ y_pos  # [N, D]
    drift_neg = W_neg @ y_neg  # [N, D]

    V = drift_pos - drift_neg
    return V


def normalize_features(x_feat, ref_feat, eps=1e-8):
    """
    Normalize features so average pairwise distance ≈ sqrt(D).
    Section A.6 of the Drifting paper.

    Args:
        x_feat: [N, D] generated sample features
        ref_feat: [M, D] reference features (positives) for computing scale

    Returns:
        scale: scalar normalization factor (detached)
    """
    D = x_feat.shape[-1]
    with torch.no_grad():
        all_feat = torch.cat([x_feat, ref_feat], dim=0)
        dist = torch.cdist(all_feat, all_feat, p=2)
        mask = ~torch.eye(all_feat.shape[0], dtype=torch.bool, device=all_feat.device)
        avg_dist = dist[mask].mean()
        scale = avg_dist / (D ** 0.5 + eps)
        scale = scale.clamp(min=eps)
    return scale


def normalize_drift(V, eps=1e-8):
    """
    Normalize drift so E[||V||²/D] ≈ 1.
    Section A.6 of the Drifting paper.

    Args:
        V: [N, D] drift vectors

    Returns:
        lambda_scale: scalar normalization factor (detached)
    """
    D = V.shape[-1]
    with torch.no_grad():
        avg_sq = (V ** 2).sum(dim=-1).mean() / D
        lambda_scale = (avg_sq + eps) ** 0.5
    return lambda_scale


def drifting_loss_single_scale(x_feat, pos_feat, neg_feat,
                                temperatures=(0.02, 0.05, 0.2),
                                self_neg=True):
    """
    Compute drifting loss for a single feature scale with normalization.

    Args:
        x_feat: [N, D] features of generated samples (WITH gradients)
        pos_feat: [N_pos, D] features of positive samples (detached)
        neg_feat: [N_neg, D] features of negative samples (detached)
        temperatures: tuple of temperature values
        self_neg: whether neg samples include x itself

    Returns:
        loss: scalar drifting loss
    """
    D = x_feat.shape[-1]
    if D == 0:
        return torch.tensor(0.0, device=x_feat.device)

    # Feature normalization (Section A.6)
    feat_scale = normalize_features(x_feat.detach(), pos_feat)
    x_norm = x_feat / feat_scale
    pos_norm = pos_feat / feat_scale
    neg_norm = neg_feat / feat_scale

    # Compute drift for each temperature and accumulate
    V_total = torch.zeros_like(x_norm)
    for tau in temperatures:
        # Scale temperature by sqrt(D) as per paper (Section A.6, Eq. 22)
        tau_scaled = tau * (D ** 0.5)
        with torch.no_grad():
            V_tau = compute_drifting_field(
                x_norm.detach(), pos_norm, neg_norm, tau_scaled, self_neg
            )
        V_total = V_total + V_tau

    # Drift normalization (Section A.6)
    drift_scale = normalize_drift(V_total)
    V_normalized = V_total / (drift_scale + 1e-8)

    # Drifting loss: ||φ(x) - stopgrad(φ(x) + V)||²
    # Gradient flows through x_norm (→ generator), target is frozen
    target = (x_norm + V_normalized).detach()
    loss = F.mse_loss(x_norm, target)

    return loss


def create_mel_filterbank(n_freqs, sr=16000, n_mels=80, fmin=0.0, fmax=None):
    """
    Create a mel-scale filterbank matrix.

    Args:
        n_freqs: number of STFT frequency bins (n_fft//2 + 1)
        sr: sample rate
        n_mels: number of mel bands
        fmin: minimum frequency
        fmax: maximum frequency (default: sr/2)

    Returns:
        fb: [n_freqs, n_mels] mel filterbank (torch.FloatTensor)
    """
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    freqs = np.linspace(0, sr / 2.0, n_freqs)

    fb = np.zeros((n_freqs, n_mels))
    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]
        up_slope = (freqs - f_left) / (f_center - f_left + 1e-10)
        down_slope = (f_right - freqs) / (f_right - f_center + 1e-10)
        fb[:, m] = np.maximum(0, np.minimum(up_slope, down_slope))

    return torch.from_numpy(fb).float()


def extract_mel_features(spec, mel_fb):
    """
    Extract mel-spectrogram features from real-valued spec.
    Stays entirely in STFT domain — no ISTFT needed.

    Args:
        spec: [B, 2, F, T] real-valued spectrogram (real and imag channels)
        mel_fb: [F, n_mels] mel filterbank matrix

    Returns:
        features: list of [B, D_j] feature tensors
    """
    B, C, F, T = spec.shape
    n_mels = mel_fb.shape[1]
    features = []

    # Magnitude from real/imag channels: [B, F, T]
    mag = torch.sqrt(spec[:, 0] ** 2 + spec[:, 1] ** 2 + 1e-8)

    # Apply mel filterbank: [B, F, T] × [F, n_mels] → [B, n_mels, T]
    mel = torch.einsum('bft,fm->bmt', mag, mel_fb)
    log_mel = torch.log(mel + 1e-8)  # [B, n_mels, T]

    # (a) Per-band mean/std: captures spectral shape → [B, n_mels] each
    features.append(log_mel.mean(dim=2))
    features.append(log_mel.std(dim=2))

    # (b) Per-frame energy: temporal envelope via mel → [B, T] each
    features.append(log_mel.mean(dim=1))
    features.append(log_mel.std(dim=1))

    # (c) Mel patch statistics (4×4 patches on mel spectrogram)
    mel_aligned = (n_mels // 4) * 4
    T_aligned = (T // 4) * 4
    if mel_aligned > 0 and T_aligned > 0:
        patches = log_mel[:, :mel_aligned, :T_aligned]
        patches = patches.reshape(B, mel_aligned // 4, 4, T_aligned // 4, 4)
        patch_mean = patches.mean(dim=(2, 4)).reshape(B, -1)
        features.append(patch_mean)

    # (d) Delta features: temporal change per band → [B, n_mels] each
    if T > 1:
        delta = log_mel[:, :, 1:] - log_mel[:, :, :-1]  # [B, n_mels, T-1]
        features.append(delta.mean(dim=2))  # avg temporal change
        features.append(delta.std(dim=2))   # temporal variability

    return features


def extract_multiscale_features(spec, mel_fb=None):
    """
    Extract multi-scale spectrogram features for drifting loss.
    Inspired by Section A.5 of the Drifting paper (multi-scale, multi-location).

    The spectrogram is treated as a 2D "image" (frequency × time) with 2 channels
    (real, imaginary). We extract statistics at multiple scales.

    Args:
        spec: [B, 2, F, T] real-valued spectrogram (real and imag channels)

    Returns:
        features: list of [B, D_j] feature tensors at different scales
    """
    B, C, F, T = spec.shape
    features = []

    # (a) Global statistics: mean and std over spatial dims → [B, 2*C]
    global_mean = spec.mean(dim=(2, 3))  # [B, C]
    global_std = spec.std(dim=(2, 3))    # [B, C]
    features.append(torch.cat([global_mean, global_std], dim=1))

    # (b) Frequency-wise: mean/std over time for each freq bin → [B, C*F] each
    freq_mean = spec.mean(dim=3).reshape(B, -1)  # [B, C*F]
    freq_std = spec.std(dim=3).reshape(B, -1)
    features.append(freq_mean)
    features.append(freq_std)

    # (c) Time-wise: mean/std over frequency for each time frame → [B, C*T] each
    time_mean = spec.mean(dim=2).reshape(B, -1)  # [B, C*T]
    time_std = spec.std(dim=2).reshape(B, -1)
    features.append(time_mean)
    features.append(time_std)

    # (d) 4×4 patch mean → [B, C * (F//4) * (T//4)]
    Fp = F // 4 * 4
    Tp = T // 4 * 4
    if Fp > 0 and Tp > 0:
        spec_cropped = spec[:, :, :Fp, :Tp]
        patches = spec_cropped.reshape(B, C, Fp // 4, 4, Tp // 4, 4)
        patch_mean = patches.mean(dim=(3, 5)).reshape(B, -1)
        features.append(patch_mean)

    # (e) Mean of squared values per channel → [B, C]
    sq_mean = (spec ** 2).mean(dim=(2, 3))  # [B, C]
    features.append(sq_mean)

    # (f) Mel-spectrogram features (perceptually meaningful, STFT-native)
    if mel_fb is not None:
        mel_feats = extract_mel_features(spec, mel_fb.to(spec.device))
        features.extend(mel_feats)

    return features


def compute_drifting_loss(x_hat_spec, x_clean_spec, temperatures=(0.02, 0.05, 0.2),
                          mel_fb=None):
    """
    Compute multi-scale drifting loss between generated and real spectrograms.

    Args:
        x_hat_spec: [B, 2, F, T] generated spectrogram (real-valued, WITH gradients)
        x_clean_spec: [B, 2, F, T] real clean spectrogram (detached)
        temperatures: tuple of temperature values for kernel
        mel_fb: [F, n_mels] optional mel filterbank for perceptual features

    Returns:
        loss: scalar total drifting loss (sum over all scales)
    """
    # Extract multi-scale features
    x_hat_feats = extract_multiscale_features(x_hat_spec, mel_fb=mel_fb)

    with torch.no_grad():
        x_clean_feats = extract_multiscale_features(x_clean_spec, mel_fb=mel_fb)

    # Compute drifting loss per scale and sum
    total_loss = torch.tensor(0.0, device=x_hat_spec.device)
    for x_feat, pos_feat in zip(x_hat_feats, x_clean_feats):
        # Negative samples = generated samples (self-negative)
        neg_feat = x_feat.detach()
        loss_j = drifting_loss_single_scale(
            x_feat, pos_feat, neg_feat,
            temperatures=temperatures,
            self_neg=True
        )
        total_loss = total_loss + loss_j

    return total_loss
