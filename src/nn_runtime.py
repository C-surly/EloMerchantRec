# -*- coding: utf-8 -*-
"""NN 训练期的轻量运行时辅助。"""

import os


DEFAULT_TRAIN_BATCH = {
    "gru": 1024,
    "gru_x": 1024,
    "trf": 1024,
}

DEFAULT_INFER_BATCH = {
    "gru": 8192,
    "gru_x": 8192,
    "trf": 8192,
}


def _env_key(prefix: str, arch: str) -> str:
    return f"{prefix}_{arch.upper().replace('-', '_')}"


def resolve_batch_size(arch: str, kind: str = "train") -> int:
    """支持全局与分架构覆盖的 batch size 解析。"""
    if kind == "train":
        default = DEFAULT_TRAIN_BATCH[arch]
        global_key = "ELO_NN_BATCH_SIZE"
        arch_key = _env_key("ELO_NN_BATCH_SIZE", arch)
    else:
        default = DEFAULT_INFER_BATCH[arch]
        global_key = "ELO_NN_INFER_BATCH_SIZE"
        arch_key = _env_key("ELO_NN_INFER_BATCH_SIZE", arch)
    return int(os.environ.get(arch_key, os.environ.get(global_key, default)))


def infer_numpy(model, xs, st, bs: int):
    """统一推理路径,减少重复样板并使用 inference_mode。"""
    import torch

    model.eval()
    with torch.inference_mode():
        outs = [model(xs[i:i + bs], st[i:i + bs]) for i in range(0, len(xs), bs)]
    return torch.cat(outs).float().cpu().numpy()


def ensure_gru_backend(device: str, input_size: int, hidden_size: int,
                       num_layers: int = 1, dropout: float = 0.0) -> bool:
    """当前环境下 cuDNN RNN 初始化失败时自动回退到原生 CUDA kernel。"""
    import torch
    import torch.nn as nn

    if not device.startswith("cuda") or not torch.backends.cudnn.enabled:
        return False
    if os.environ.get("ELO_DISABLE_CUDNN_RNN") == "1":
        torch.backends.cudnn.enabled = False
        return True
    try:
        probe = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                       batch_first=True, dropout=dropout).to(device)
        sample = torch.zeros(2, 2, input_size, device=device)
        with torch.inference_mode():
            probe(sample)
        del sample, probe
        return False
    except RuntimeError as exc:
        if "CUDNN_STATUS_NOT_INITIALIZED" not in str(exc):
            raise
        torch.backends.cudnn.enabled = False
        return True
