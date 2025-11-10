import torch
from typing import Callable, List, Any
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


def benchmark_torch(
    fn: Callable,
    workspace_generator: Callable,
    workspace_count: int = 1,
    warmup_iterations: int = 10,
    iterations: int = 100,
):
    assert fn is not None
    assert workspace_generator is not None
    assert warmup_iterations > 0
    assert iterations > 0

    workspaces = [workspace_generator() for _ in range(workspace_count)]

    workspace_index = 0
    torch.cuda.empty_cache()
    for _ in range(warmup_iterations):
        workspace = workspaces[workspace_index]
        fn(*workspace)
        workspace_index = (workspace_index + 1) % workspace_count
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        workspace = workspaces[workspace_index]
        fn(*workspace)
        workspace_index = (workspace_index + 1) % workspace_count
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / iterations
    return avg_time_ms * 1e3  # return in microseconds
