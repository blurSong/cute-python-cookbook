import torch
from cutlass.base_dsl.typing import __STR_TO_DTYPE__

VERBOSE = False
LOG = "[CuTe Info]"


def check_cuda():
    assert torch.cuda.is_available(), "NO CUDA device detected."


def get_cutlass_dtype(type: str):
    for k, v in __STR_TO_DTYPE__.items():
        if type == k.lower():
            return v
    raise ValueError(f"Unknown type: {type}")
