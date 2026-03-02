import torch
import torch.nn.functional as F
import torchaudio
from pesq import pesq
from pystoi import stoi
import soundfile as sf

from .other import si_sdr, pad_spec
from ..sampling import get_white_box_solver

sr = 16000
N = 5


def load_audio(filepath, target_sr=16000):
    """Load audio and resample to target_sr (16kHz by default)."""
    waveform, sample_rate = sf.read(filepath)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform, target_sr


def evaluate_model(model, num_eval_files, inference_N=N):
    T_rev = model.T_rev
    model.ode.T_rev = T_rev
    t_eps = model.t_eps
    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files

    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files - 1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)

    inference_N = inference_N
    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for clean_file, noisy_file in zip(clean_files, noisy_files):
        # Load wavs
        x, _ = load_audio(clean_file)
        y, _ = load_audio(noisy_file)
        T_orig = x.size(1)

        # Normalize per utterance
        norm_factor = y.abs().max()
        y = y / norm_factor

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)

        y = y * norm_factor

        # Reverse sampling
        sampler = get_white_box_solver(
            "euler", model.ode, model, Y.cuda(), T_rev=T_rev, t_eps=t_eps, N=inference_N
        )

        sample, _ = sampler()

        sample = sample.squeeze()

        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        _si_sdr += si_sdr(x, x_hat)

        _pesq += pesq(sr, x, x_hat, "wb")
        _estoi += stoi(x, x_hat, sr, extended=True)

    return _pesq / num_eval_files, _si_sdr / num_eval_files, _estoi / num_eval_files
