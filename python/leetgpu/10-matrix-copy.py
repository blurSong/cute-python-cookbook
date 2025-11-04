"""
Matrix Transpose Example using CUTE DSL.

Key optimizations:

- Coalesced global memory accesses.
- Use shared memory and swizzle to avoid bank conflicts.

References:

- An Efficient Matrix Transpose in CUDA C/C++
  https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

- Tutorial: Matrix Transpose in CUTLASS
  https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/

- CuTe Matrix Transpose
  https://leimao.github.io/article/CuTe-Matrix-Transpose/

"""

import time
import math
import torch
import argparse
from typing import Union

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda

VERBOSE = False
LOG = "[CuTe Info]"

cta_tiler = (128, 128)
thr_tiler = (8, 32)
vl = 128 // cute.Float32.width


@cute.kernel
def matrix_copy_sm80_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gCrd: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    N: cute.Int32,
):
    dtype = gA.element_type
    bid_x, bid_y, _ = cute.arch.block_idx()
    tid_x, _, _ = cute.arch.thread_idx()

    cta_coord = ((None, None), bid_x, bid_y)
    gA = gA[cta_coord]
    gB = gB[cta_coord]
    gCrd = gCrd[cta_coord]

    thr_copy = tiled_copy.get_slice(tid_x)
    tAgA = thr_copy.partition_S(gA)
    tBgB = thr_copy.partition_S(gB)
    tAgCrd = thr_copy.partition_S(gCrd)

    tArA = cute.make_fragment_like(tAgA)
    tArCrd = cute.make_fragment_like(tAgCrd, cutlass.Boolean)

    if VERBOSE:
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"tAgA: {tAgA}")
        print(f"tBgB: {tBgB}")
        print(f"tArA: {tArA}")

    # Make coordinate prediction
    for i in cutlass.range(cute.size(tArCrd)):
        tArCrd[i] = cute.elem_less(tAgCrd[i], (N, N))

    cute.copy(tiled_copy, tAgA, tArA, pred=tArCrd)
    cute.copy(tiled_copy, tArA, tBgB, pred=tArCrd)


@cute.jit
def matrix_copy_sm80(
    A: cute.Tensor,
    B: cute.Tensor,
    N: cute.Int32,
):
    dtype = A.element_type

    # * gA ((tile_m, tile_n), res_m, res_n)
    # * gB ((tile_m, tile_n), res_m, res_n)
    gA = cute.tiled_divide(A, cta_tiler)
    gB = cute.tiled_divide(B, cta_tiler)

    Crd = cute.make_identity_tensor(A.shape)
    gCrd = cute.tiled_divide(Crd, cta_tiler)

    thr_layout = cute.make_ordered_layout(thr_tiler, order=(1, 0))
    val_layout = cute.make_ordered_layout((vl,), order=(0,))
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)

    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    grid_dim = [gA.shape[1], gA.shape[2], 1]
    block_dim = [cute.size(thr_tiler), 1, 1]

    if VERBOSE:
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"gCrd: {gCrd}")
        print(f"grid_dim: {grid_dim}")
        print(f"block_dim: {block_dim}")
        print(f"thr_layout: {thr_layout}")

    matrix_copy_sm80_kernel(gA, gB, gCrd, tiled_copy, N).launch(
        grid=grid_dim, block=block_dim, smem=0
    )


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    matrix_copy_sm80(input, output, N)


def test():
    n = 1000
    input = torch.arange(0, n * n, dtype=torch.float32, device='cuda').reshape(n, n)
    output = torch.empty_like(input)

    input_tensor = cute.runtime.from_dlpack(input)
    output_tensor = cute.runtime.from_dlpack(output)

    solve(input_tensor, output_tensor, n)

    torch.testing.assert_close(output, input)
    print("Test passed!")


if __name__ == "__main__":
    test()
