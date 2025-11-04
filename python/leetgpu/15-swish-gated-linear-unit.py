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
def swiglu_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    tiled_copy_input: cute.TiledCopy,
    N_halved: cute.Int32,
):
    tid_x = cute.arch.thread_idx()[0]
    bid_x = cute.arch.block_idx()[0]

    input_tile_0 = input[(None, bid_x, 0)]
    input_tile_1 = input[(None, bid_x, 1)]
    output_tile = output[(None, bid_x)]

    thr_copy_input = tiled_copy_input.get_slice(tid_x)
    input_thr_0 = thr_copy_input.partition_S(input_tile_0)
    input_thr_1 = thr_copy_input.partition_S(input_tile_1)
    output_thr = thr_copy_input.partition_S(output_tile)

    input_frag_thr_0 = cute.make_fragment_like(input_thr_0)
    input_frag_thr_1 = cute.make_fragment_like(input_thr_1)
    output_frag_thr = cute.make_fragment_like(output_thr)

    if VERBOSE:
        print(f"{LOG} input_tile_0/1", input_tile_0)
        print(f"{LOG} output_tile", output_tile)
        print(f"{LOG} input_thr_0/1", input_frag_thr_0)
        print(f"{LOG} output_thr", output_thr)

    # Boundary check
    thr_offset = tid_x * vl + bid_x * threads * vl
    thr_frag_pred = cute.make_fragment((vl,), cutlass.Boolean)
    for i in range(vl):
        thr_frag_pred[i] = cute.elem_less(i + thr_offset, N_halved)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)
    cute.copy(copy_atom, input_thr_0[None, 0], input_frag_thr_0[None, 0], pred=thr_frag_pred)
    cute.copy(copy_atom, input_thr_1[None, 0], input_frag_thr_1[None, 0], pred=thr_frag_pred)
    output_frag_thr.store(swiglu(input_frag_thr_0.load(), input_frag_thr_1.load()))
    cute.copy(copy_atom, output_frag_thr[None, 0], output_thr[None, 0], pred=thr_frag_pred)


# A, B, C are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    N_halved = N // 2
    thr_layout = cute.make_layout((threads,))
    val_layout = cute.make_layout((vl,))
    tiler_n, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    blocks = cute.size(cute.ceil_div(N_halved, tiler_n))

    input_tiled = cute.make_tensor(
        input.iterator,
        layout=cute.make_layout((tiler_n[0], blocks, 2), stride=(1, tiler_n[0], N_halved)),
    )
    output_tiled = cute.flat_divide(output, tiler_n)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    grid_size = [blocks, 1, 1]
    block_size = [threads, 1, 1]

    if VERBOSE:
        print(f"{LOG} input: {input_tiled}")
        print(f"{LOG} tv_layout: {tv_layout}")
        print(f"{LOG} tiler: {tiler_n}")
        print(f"{LOG} grid_size: {grid_size}")
        print(f"{LOG} block_size: {block_size}")

    swiglu_kernel(input_tiled, output_tiled, tiled_copy, N_halved).launch(
        grid=grid_size, block=block_size
    )


def test():
    n = 100
    input = torch.randint(-100, 100, (n,), dtype=torch.float32, device='cuda')
    output = torch.empty(n // 2, dtype=torch.float32, device='cuda')

    input_tensor = cute.runtime.from_dlpack(input)
    output_tensor = cute.runtime.from_dlpack(output)

    solve(input_tensor, output_tensor, n)

    x1, x2 = input.chunk(2)
    output_ref = (x1 * torch.sigmoid(x1)) * x2
    torch.testing.assert_close(output, output_ref)
    print("Test passed!")


if __name__ == "__main__":
    test()
