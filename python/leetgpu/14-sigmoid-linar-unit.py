import numpy
import torch

import cutlass
import cutlass.cute as cute

# NOTE in cutlass 4.3 only import cutlass.dsl_user_op
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op

VERBOSE = False
LOG = "[CuTe Info][LeetGPU]"

threads = 256
vl = 128 // cute.Float32.width


@dsl_user_op
def silu(x: cute.TensorSSA, *, loc=None, ip=None):
    exp_mius_x = cute.exp(-x)
    return x / (1 + exp_mius_x)


@dsl_user_op
def swiglu(x: cute.TensorSSA, y: cute.TensorSSA, *, loc=None, ip=None):
    return silu(x, loc=loc, ip=ip) * y


@cute.kernel
def silu_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    tiled_copy_input: cute.TiledCopy,
    N: cute.Int32,
):
    tid_x = cute.arch.thread_idx()[0]
    bid_x = cute.arch.block_idx()[0]

    input_tile = input[(None, bid_x)]
    output_tile = output[(None, bid_x)]

    thr_copy_input = tiled_copy_input.get_slice(tid_x)
    input_thr = thr_copy_input.partition_S(input_tile)
    output_thr = thr_copy_input.partition_S(output_tile)

    input_frag_thr = cute.make_fragment_like(input_thr)
    output_frag_thr = cute.make_fragment_like(output_thr)

    if VERBOSE:
        print(f"{LOG} input_tile", input_tile)
        print(f"{LOG} output_tile", output_tile)
        print(f"{LOG} input_thr", input_thr)
        print(f"{LOG} output_thr", output_thr)

    # Boundary check
    thr_offset = tid_x * vl + bid_x * threads * vl
    thr_frag_pred = cute.make_fragment((vl,), cutlass.Boolean)
    for i in range(vl):
        thr_frag_pred[i] = cute.elem_less(i + thr_offset, N)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)
    cute.copy(copy_atom, input_thr[None, 0], input_frag_thr[None, 0], pred=thr_frag_pred)
    output_frag_thr.store(silu(input_frag_thr.load()))
    cute.copy(copy_atom, output_frag_thr[None, 0], output_thr[None, 0], pred=thr_frag_pred)


# A, B, C are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    thr_layout = cute.make_layout((threads,))
    val_layout = cute.make_layout((vl,))
    tiler_n, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    input = cute.flat_divide(input, tiler_n)  # (threads * vl, res_n)
    output = cute.flat_divide(output, tiler_n)  # (threads * vl, res_n)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    grid_size = [cute.ceil_div(N, tiler_n), 1, 1]
    block_size = [threads, 1, 1]

    if VERBOSE:
        print(f"{LOG} input: {input}")
        print(f"{LOG} tv_layout: {tv_layout}")
        print(f"{LOG} tiler: {tiler_n}")
        print(f"{LOG} grid_size: {grid_size}")
        print(f"{LOG} block_size: {block_size}")

    silu_kernel(input, output, tiled_copy, N).launch(grid=grid_size, block=block_size)


def test():
    n = 1000
    input = torch.randint(-1000, 1000, (n,), dtype=torch.float32, device='cuda')
    output = torch.empty_like(input)

    input_tensor = cute.runtime.from_dlpack(input)
    output_tensor = cute.runtime.from_dlpack(output)

    solve(input_tensor, output_tensor, n)

    torch.testing.assert_close(output, input * torch.sigmoid(input))
    print("Test passed!")


if __name__ == "__main__":
    test()
