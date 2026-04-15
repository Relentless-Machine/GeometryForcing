from .third_party.common_metrics_on_video_quality.calculate_lpips import loss_fn as img_lpips_loss_fn
from .third_party.common_metrics_on_video_quality.calculate_psnr import img_psnr
from .third_party.common_metrics_on_video_quality.calculate_ssim import calculate_ssim_function

# fallback methods

# from __future__ import annotations

# from typing import Optional

# import numpy as np
# import torch

# try:
#     from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
# except Exception:  # pragma: no cover - fallback for older torchmetrics layouts
#     peak_signal_noise_ratio = None
#     structural_similarity_index_measure = None

# try:
#     from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# except Exception:  # pragma: no cover - keep evaluation runnable if LPIPS is unavailable
#     LearnedPerceptualImagePatchSimilarity = None


# def _as_nchw_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
#     tensor = torch.as_tensor(image)
#     if tensor.ndim != 3:
#         raise ValueError(f"Expected 3D image tensor, got shape {tuple(tensor.shape)}")

#     if tensor.shape[0] in (1, 3) and tensor.shape[-1] not in (1, 3):
#         tensor = tensor
#     else:
#         tensor = tensor.permute(2, 0, 1)

#     tensor = tensor.to(dtype=torch.float32)
#     if tensor.max().item() > 1.5:
#         tensor = tensor / 255.0
#     return tensor.unsqueeze(0)


# def img_psnr(preds: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
#     preds_tensor = _as_nchw_tensor(preds)
#     target_tensor = _as_nchw_tensor(target)
#     if peak_signal_noise_ratio is not None:
#         return float(peak_signal_noise_ratio(preds_tensor, target_tensor, data_range=1.0).item())
#     mse = torch.mean((preds_tensor - target_tensor) ** 2)
#     if mse.item() == 0:
#         return float("inf")
#     return float(20.0 * torch.log10(torch.tensor(1.0)) - 10.0 * torch.log10(mse).item())


# def calculate_ssim_function(preds: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
#     preds_tensor = _as_nchw_tensor(preds)
#     target_tensor = _as_nchw_tensor(target)
#     if structural_similarity_index_measure is not None:
#         return float(structural_similarity_index_measure(preds_tensor, target_tensor, data_range=1.0).item())
#     raise ModuleNotFoundError("torchmetrics.structural_similarity_index_measure is unavailable")


# if LearnedPerceptualImagePatchSimilarity is not None:
#     img_lpips_loss_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False)
# else:
#     img_lpips_loss_fn = None
