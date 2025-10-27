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
from utils import *

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda

VERBOSE = False
LOG = "[CuTe Info]"

cta_tiler = (64, 64)
thr_tiler = (8, 32)


@cute.kernel
def transpose_naive_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gCrd: cute.Tensor,
    thr_layout: cute.Layout,
    global_shape: cutlass.Shape,
):
    dtype = gA.element_type
    bid_x, bid_y, _ = cute.arch.block_idx()
    tid_x, _, _ = cute.arch.thread_idx()

    cta_coord = ((None, None), bid_x, bid_y)
    gA = gA[cta_coord]
    gB = gB[cta_coord]
    tAgA = cute.local_partition(gA, thr_layout, tid_x)
    tBgB = cute.local_partition(gB, thr_layout, tid_x)

    tArA = cute.make_fragment_like(tAgA)

    if VERBOSE:
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"tAgA: {tAgA}")
        print(f"tBgB: {tBgB}")
        print(f"tArA: {tArA}")

    # Make coordinate prediction
    gCrd = gCrd[cta_coord]
    tAgCrd = cute.local_partition(gCrd, thr_layout, tid_x)
    tArCrd = cute.make_fragment_like(tAgCrd, cutlass.Boolean)

    for i in range(cute.size(tArCrd), unroll=1):
        tArCrd[i] = cute.elem_less(tAgCrd[i], global_shape)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)

    cute.copy(copy_atom, tAgA, tArA, pred=tArCrd)
    cute.copy(copy_atom, tArA, tBgB, pred=tArCrd)


@cute.kernel
def transpose_smem_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gCrd: cute.Tensor,
    thr_layout_a: cute.Layout,
    thr_layout_b: cute.Layout,
    smem_layout_a: Union[cute.Layout, cute.ComposedLayout],
    smem_layout_b: Union[cute.Layout, cute.ComposedLayout],
    global_shape: cutlass.Shape,
):
    dtype = gA.element_type
    bid_x, bid_y, _ = cute.arch.block_idx()
    tid_x, _, _ = cute.arch.thread_idx()

    # Note: can also define swizzle here using allocate_tensor()
    smem_ptr = cutlass.utils.SmemAllocator().allocate_array(dtype, cute.cosize(smem_layout_a))

    sA = cute.make_tensor(smem_ptr, smem_layout_a)
    sB = cute.make_tensor(smem_ptr, smem_layout_b)

    cta_coord = ((None, None), bid_x, bid_y)
    gA = gA[cta_coord]
    gB = gB[cta_coord]

    tAgA = cute.local_partition(gA, thr_layout_a, tid_x)
    tBgB = cute.local_partition(gB, thr_layout_b, tid_x)
    tAsA = cute.local_partition(sA, thr_layout_a, tid_x)
    tBsB = cute.local_partition(sB, thr_layout_b, tid_x)

    if VERBOSE:
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"tAgA: {tAgA}")
        print(f"tBgB: {tBgB}")
        print(f"tAsA: {tAsA}")
        print(f"tBsB: {tBsB}")

    # Make coordinate prediction
    gCrd = gCrd[cta_coord]
    tAgCrd = cute.local_partition(gCrd, thr_layout_a, tid_x)
    tBgCrd = cute.local_partition(gCrd, thr_layout_b, tid_x)
    tArCrd = cute.make_fragment_like(tAgCrd, cutlass.Boolean)
    tBrCrd = cute.make_fragment_like(tBgCrd, cutlass.Boolean)

    for i in range(cute.size(tArCrd), unroll=1):
        tArCrd[i] = cute.elem_less(tAgCrd[i], global_shape)
    for i in range(cute.size(tBrCrd), unroll=1):
        tBrCrd[i] = cute.elem_less(tBgCrd[i], global_shape)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)

    cute.copy(copy_atom, tAgA, tAsA, pred=tArCrd)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()
    cute.copy(copy_atom, tBsB, tBgB, pred=tBrCrd)


@cute.jit
def transpose_naive(
    A: cute.Tensor,
    B: cute.Tensor,
    stream: cuda.CUstream,
    copy_bits: cutlass.Constexpr = 128,
):
    # War. For output matrix B, we define a column-major layout (M,N):(1,M)
    B = cute.make_tensor(B.iterator, cute.make_layout(B.shape, stride=(1, B.shape[0])))

    dtype = A.element_type
    vl = copy_bits // dtype.width

    a_major_mode = utils.LayoutEnum.from_tensor(A)
    b_major_mode = utils.LayoutEnum.from_tensor(B)
    assert a_major_mode == utils.LayoutEnum.ROW_MAJOR
    assert b_major_mode == utils.LayoutEnum.COL_MAJOR

    # * gA ((tile_m, tile_n), res_m, res_n)
    # * gB ((tile_m, tile_n), res_m, res_n)
    gA = cute.tiled_divide(A, cta_tiler)
    gB = cute.tiled_divide(B, cta_tiler)

    Crd = cute.make_identity_tensor(A.shape)
    gCrd = cute.tiled_divide(Crd, cta_tiler)

    # - Here we define 2 thread-level layouts for coalesced access
    #   Either use (8,32) for A coalesced or (32,8) for B coalesced
    # - Based on empirical testing, B coalesced achieves better performance
    thr_layout_a_coalesced = cute.make_ordered_layout(thr_tiler, order=(1, 0))
    thr_layout_b_coalesced = cute.make_ordered_layout(thr_tiler, order=(0, 1))
    thr_layout = thr_layout_b_coalesced

    grid_dim = [cute.size(gA, mode=[1]), cute.size(gA, mode=[2]), 1]
    block_dim = [cute.size(thr_tiler), 1, 1]

    if VERBOSE:
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"gCrd: {gCrd}")
        print(f"grid_dim: {grid_dim}")
        print(f"block_dim: {block_dim}")
        print(f"thr_layout: {thr_layout}")

    transpose_naive_kernel(gA, gB, gCrd, thr_layout, A.shape).launch(
        grid=grid_dim, block=block_dim, smem=0, stream=stream
    )


@cute.jit
def transpose_smem(
    A: cute.Tensor,
    B: cute.Tensor,
    stream: cuda.CUstream,
    copy_bits: cutlass.Constexpr = 128,
    swizzle: cutlass.Constexpr = True,
):
    # War. For output matrix B, we define a column-major layout (M,N):(1,M)
    B = cute.make_tensor(B.iterator, cute.make_layout(B.shape, stride=(1, B.shape[0])))

    dtype = A.element_type
    vl = copy_bits // dtype.width

    a_major_mode = utils.LayoutEnum.from_tensor(A)
    b_major_mode = utils.LayoutEnum.from_tensor(B)
    assert a_major_mode == utils.LayoutEnum.ROW_MAJOR
    assert b_major_mode == utils.LayoutEnum.COL_MAJOR

    # * gA ((tile_m, tile_n), res_m, res_n)
    # * gB ((tile_m, tile_n), res_m, res_n)
    gA = cute.tiled_divide(A, cta_tiler)
    gB = cute.tiled_divide(B, cta_tiler)

    Crd = cute.make_identity_tensor(A.shape)
    gCrd = cute.tiled_divide(Crd, cta_tiler)

    thr_layout_a_coalesced = cute.make_ordered_layout(thr_tiler, order=(1, 0))
    thr_layout_b_coalesced = cute.make_ordered_layout(thr_tiler, order=(0, 1))

    smem_layout_a_coalesced = cute.make_ordered_layout(cta_tiler, order=(1, 0))
    smem_layout_b_coalesced = cute.make_ordered_layout(cta_tiler, order=(0, 1))
    smem_layout_a_swizzled, smem_layout_b_swizzled = None, None

    if cutlass.const_expr(swizzle):
        m = math.log(1, 2)
        b = math.log(32 * 32 / dtype.width, 2) - m  # 32 is bank number
        s = math.log(cta_tiler[1], 2) - m
        swizzle_atom = cute.make_swizzle(b, m, s)
        smem_layout_a_swizzled = cute.make_composed_layout(swizzle_atom, 0, smem_layout_a_coalesced)
        smem_layout_b_swizzled = cute.composition(smem_layout_a_swizzled, smem_layout_b_coalesced)
        # R=lhs(rhs(c))

    # Here we row-major coalesced access for A and column-major coalesced access for B
    # Applied for both shared memory and thread-level layouts
    thr_layout_a = thr_layout_a_coalesced
    thr_layout_b = thr_layout_b_coalesced
    if cutlass.const_expr(swizzle):
        smem_layout_a = smem_layout_a_swizzled
        smem_layout_b = smem_layout_b_swizzled
    else:
        smem_layout_a = smem_layout_a_coalesced
        smem_layout_b = smem_layout_b_coalesced

    grid_dim = [cute.size(gA, mode=[1]), cute.size(gA, mode=[2]), 1]
    block_dim = [cute.size(thr_tiler), 1, 1]
    smem_size = cute.cosize(smem_layout_a) * dtype.width // 8

    if VERBOSE:
        print(f"A: {A}")
        print(f"B: {B}")
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"grid_dim: {grid_dim}")
        print(f"block_dim: {block_dim}")
        print(f"smem_layout_a: {smem_layout_a}")
        print(f"smem_layout_b: {smem_layout_b}")
        print(f"thr_layout_a: {thr_layout_a}")
        print(f"thr_layout_b: {thr_layout_b}")

    transpose_smem_kernel(
        gA, gB, gCrd, thr_layout_a, thr_layout_b, smem_layout_a, smem_layout_b, A.shape
    ).launch(grid=grid_dim, block=block_dim, smem=smem_size, stream=stream)


def run_transpose(
    impl: str,
    M: int,
    N: int,
    dtype: cutlass.Numeric = cutlass.Float32,
    warmup_iterations: int = 10,
    iterations: int = 100,
    dynamic_layout: bool = False,
    skip_verify: bool = False,
):
    print(f"Running transpose with M={M}, N={N}, dtype={dtype}, impl={impl}")

    torch_stream = torch.cuda.Stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    torch_dtype = cutlass_torch.dtype(dtype)

    kernel = None
    kwargs = {}
    if impl == "naive":
        kernel = transpose_naive
    elif impl == "smem":
        kernel = transpose_smem
    elif impl == "smem-swizzle":
        kernel = transpose_smem
        kwargs = {"swizzle": True}
    else:
        raise ValueError(f"Unsupported impl: {impl}")

    def _tensor_generator(return_torch=False):
        a = torch.arange(M * N).view(M, N).to(dtype=torch_dtype).to("cuda")
        b = torch.empty((M, N)).to(dtype=torch_dtype).to("cuda")

        a_tensor = from_dlpack(a) if not dynamic_layout else from_dlpack(a).mark_layout_dynamic()
        b_tensor = from_dlpack(b) if not dynamic_layout else from_dlpack(b).mark_layout_dynamic()

        if return_torch:
            return a_tensor, b_tensor, a, b

        return a_tensor, b_tensor

    if not skip_verify:
        _a_tensor, _b_tensor, _a, _b = _tensor_generator(True)
        kernel(_a_tensor, _b_tensor, current_stream, **kwargs)
        print(_a, "\n", _b.reshape(N, M))
        torch.testing.assert_close(_a.T, _b.reshape(N, M))
        print("Transpose correctness check passed.")
    else:
        print("Transpose correctness check skipped.")

    # Compile
    compile_tic = time.perf_counter()
    transpose_func = cute.compile(kernel, *_tensor_generator(), current_stream, **kwargs)
    print(f"Kernel compiled in {time.perf_counter() - compile_tic:.4f} seconds")

    # Benchmarking
    torch.cuda.empty_cache()
    workspace_bytes = M * N * dtype.width // 8
    workspace_count = testing.get_workspace_count(workspace_bytes, warmup_iterations, iterations)
    workspace_generator = lambda: testing.JitArguments(*_tensor_generator(), current_stream)

    average_kernel_time_us = testing.benchmark(
        transpose_func,
        workspace_generator=workspace_generator,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        stream=current_stream,
        use_cuda_graphs=False,
    )

    print(f"Kernel execution time: {average_kernel_time_us / 1e3:.4f} ms")
    print(
        f"Achieved memory throughput: {(2 * M * N * dtype.width // 8) / (average_kernel_time_us / 1e6) / 1e9:.2f} GB/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise ops to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument(
        "--impl", type=str, choices=["naive", "smem", "smem-swizzle", "tma"], default="naive"
    )
    parser.add_argument("--M", "-M", default=1024, type=int)
    parser.add_argument("--N", "-N", default=1024, type=int)
    parser.add_argument("--dtype", default="int32", type=str)
    parser.add_argument("--warmup-iterations", default=5, type=int)
    parser.add_argument("--iterations", default=30, type=int)
    parser.add_argument("--dynamic-layout", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    check_cuda()

    VERBOSE = args.verbose

    run_transpose(
        args.impl,
        args.M,
        args.N,
        dtype=get_cutlass_dtype(args.dtype),
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        dynamic_layout=args.dynamic_layout,
        skip_verify=args.skip_verify,
    )
    print("PASS!")
