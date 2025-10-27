"""
A dense FP32 SIMT GEMM using CUTE DSL.

The SGEMM implementation handles only row-major or col-major layouts.
To bridge the gap of GEMM order between BLAS and CUTE, we can use the following layouts.
------------------------------------------
Blas      T                   N
------------------------------------------
A         (M, K):(K, 1)      (M, K):(1, M)
B         (N, K):(1, N)      (N, K):(K, 1)
------------------------------------------
See also:
    https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.html#aside-m-major-n-major-k-major


This GEMM kernel supports the following features:
    - Utilizes FPU for matrix multiply-accumulate (MMA) operations
    - Use multistage pipeline to overlap computation and memory access
      * Shared memory pipeline: hides gmem-to-smem latency.
      * Register pipeline: overlaps shared memory-to-register transfers with
        computations and eliminates false data dependencies for
        better parallelism.
    - Use vectorized copies
    - Add padding to reduce bank conflicts in global -> shared memory copies
    - Use predication to avoid unnecessary copies or copies of stale data

Blog: CUTLASS: Fast Linear Algebra in CUDA C++
    https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/


To run this example:

.. code-block:: bash

    python sgemm_tiling.py -M 8192 -N 8192 -K 8192

To collect performance with NCU profiler:

.. code-block:: bash

    ncu -o ../../logs/sgemm_tiling -f --set full \
        python sgemm_tiling.py \
        --skip-verify --warmup-iterations 1  --iterations 1


Constraints:
    1. Supported input, output, and accumulator data types: fp32
    2. Default tile shape is set to be 128x128x8
    3. The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned
"""

import time
import os
import argparse
import operator
from typing import Type, List, Tuple

import torch
import numpy

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack


VERBOSE = False
LOG = "[CuTe Info]"

num_stages = 3
cta_tiler: Tuple[int, int, int] = (128, 128, 8)
mma_tiler: Tuple[int, int] = (16, 16)


def check_cuda():
    assert torch.cuda.is_available(), "NO CUDA device detected."


@cute.kernel
def sgemm_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    smem_layout_a: cute.Layout,
    smem_layout_b: cute.Layout,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    tid_x, _, _ = cute.arch.thread_idx()
    bid_x, bid_y, _ = cute.arch.block_idx()

    # Use local tile to tile the gA/B/C.
    # - The grid we use is (M/tile_m, N/tile_n), so each cta handles
    #   one (tile_m, K)@A and one (tile_n, K)@B
    # - Another choice is to use zipped_devide and apply the coord for indexing.
    #   https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.html#cta-partitioning

    tiler_coord = (bid_x, bid_y, None)
    # Here None means 'all' for cute slicing

    # * gA (tile_m, tile_k, num_tiles_k)
    # * gB (tile_n, tile_k, num_tiles_k)
    # * gC (tile_m, tile_n)
    gA = cute.local_tile(A, cta_tiler, tiler_coord, proj=(1, None, 1))
    gB = cute.local_tile(B, cta_tiler, tiler_coord, proj=(None, 1, 1))
    gC = cute.local_tile(C, cta_tiler, tiler_coord, proj=(1, 1, None))
    # But here, None means 'not select' as X for Cute C++ API.

    if VERBOSE:
        print(f"{LOG} gA {gA.type}")
        print(f"{LOG} gB {gB.type}")
        print(f"{LOG} gC {gC.type}")

    # - Optimization: Move the pointer of gA/gB in the -k direction, making the
    #   first tile (instead of the last one) irregular in shape when k is irregular.
    # - We first handle the irregular tile in the prologue
    #   to avoid checking for this condition within the mainloop.
    # * copies (atom_v, rest_v) # i.e., the val_layout.
    # * tAgA (copies, copy_m, copy_k, num_tiles_k)
    # * tAsA (copies, copy_m, copy_k, num_stages)
    # * tBgB (copies, copy_n, copy_k, num_tiles_k)
    # * tBsB (copies, copy_n, copy_k, num_stages)
    residue_k = A.shape[1] - cutlass.Int32(cta_tiler[2]) * gA.shape[2]
    gA = cute.domain_offset((0, residue_k, 0), gA)
    gB = cute.domain_offset((0, residue_k, 0), gB)

    # Get the thread-level tiles
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, smem_layout_a, 16)
    sB = smem.allocate_tensor(cutlass.Float32, smem_layout_b, 16)
    thr_copy_a = tiled_copy_a.get_slice(tid_x)
    thr_copy_b = tiled_copy_b.get_slice(tid_x)
    tAgA = thr_copy_a.partition_S(gA)
    tAsA = thr_copy_a.partition_D(sA)
    tBgB = thr_copy_b.partition_S(gB)
    tBsB = thr_copy_b.partition_D(sB)

    if VERBOSE:
        print(f"{LOG} tAgA {tAgA}")
        print(f"{LOG} tAsA {tAsA}")
        print(f"{LOG} tBgB {tBgB}")
        print(f"{LOG} tBsB {tBsB}")

    thr_mma = tiled_mma.get_slice(tid_x)  # here use 1d tid_x to index 2d mma.

    # Predication: Mark indices that need to copy when the problem shape isn't
    # a multiple of the tile shape.
    # - If tApredA/B[i] is 0, then do not do the copy
    #   atom associated with index i.
    # - Refer to https://shorturl.at/8D9PK

    # Construct identity layout for sA and sB, used for predication
    # * crdA (M_div_m, K_div_k) o (tile_m, tile_k, num_tiles_k)
    # * crdB (K_div_n, N_div_n) o (tile_n, tile_k, num_tiles_k)
    # * tAcrdA (K_div_n, N_div_n) o (copies, copy_m, copy_k, num_tiles_k)
    # * tBcrdB (M_div_m, K_div_k) o (copies, copy_n, copy_k, num_tiles_k)
    # * tAprdA (rest_v, copy_m, copy_k):(copy_m, 1, 0)
    # * tAprdB (rest_v, copy_n, copy_k):(copy_n, 1, 0)
    crdA = cute.make_identity_tensor(A.shape)
    crdB = cute.make_identity_tensor(B.shape)
    crdA = cute.local_tile(crdA, cta_tiler, tiler_coord, proj=(1, None, 1))
    crdB = cute.local_tile(crdB, cta_tiler, tiler_coord, proj=(None, 1, 1))
    crdA = cute.domain_offset((0, residue_k, 0), crdA)
    crdB = cute.domain_offset((0, residue_k, 0), crdB)
    tAcrdA = thr_copy_a.partition_S(crdA)
    tBcrdB = thr_copy_b.partition_S(crdB)

    if VERBOSE:
        print(f"{LOG} crdA {crdA}")
        print(f"{LOG} crdB {crdB}")
        print(f"{LOG} tAcrdA {tAcrdA}")
        print(f"{LOG} tBcrdB {tBcrdB}")

    # Allocate predicate fragments.
    # - We allocate two kinds of pred frags. One for main-loop without residual,
    #   one for the residual tile of k.
    # - Here the mode3's stride is 0. We only pred the m and n bounds. k is broadcasted.
    tAprdA = cute.make_fragment(
        cute.make_layout(
            shape=(tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
            stride=(cute.size(tAsA, mode=[1]), 1, 0),
        ),
        cutlass.Boolean,
    )
    tBprdB = cute.make_fragment(
        cute.make_layout(
            shape=(tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
            stride=(cute.size(tBsB, mode=[1]), 1, 0),
        ),
        cutlass.Boolean,
    )
    tAprdA_res_k = cute.make_fragment(
        cute.make_layout(
            shape=(tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
            stride=(
                cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                cute.size(tAsA, mode=[2]),
                1,
            ),
        ),
        cutlass.Boolean,
    )
    tBprdB_res_k = cute.make_fragment(
        cute.make_layout(
            shape=(tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
            stride=(
                cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                cute.size(tBsB, mode=[2]),
                1,
            ),
        ),
        cutlass.Boolean,
    )

    # Set predicates for m/n bounds for mainloop and m/n/k bounds for residue k tile.
    # - 0-cord means broadcast. For tAprdA and tAprdB, k is always within the bounds. So we
    #   only compare the m- and n-coordinates of the 0th k-tile and 0th k-block.
    # - The stride-0 broadcasting mode still allows us to treat this data as a predicate tensor
    #   for each and every element of the tile to be loaded.
    for rest_v in range(tAprdA.shape[0], unroll_full=True):
        for m in range(tAprdA.shape[1], unroll_full=True):
            tAprdA[rest_v, m, 0] = cute.elem_less(tAcrdA[(0, rest_v), m, 0, 0][0], A.shape[0])
    for rest_v in range(tBprdB.shape[0], unroll_full=True):
        for n in range(tBprdB.shape[1], unroll_full=True):
            tBprdB[rest_v, n, 0] = cute.elem_less(tBcrdB[(0, rest_v), n, 0, 0][0], B.shape[0])

    for rest_v in range(tAprdA_res_k.shape[0], unroll_full=True):
        for m in range(tAprdA_res_k.shape[1], unroll_full=True):
            for k in range(tAprdA_res_k.shape[2], unroll_full=True):
                crdA_tmp = tAcrdA[(0, rest_v), m, k, 0]
                tAprdA_res_k[rest_v, m, k] = cute.elem_less(
                    (crdA_tmp[0], cutlass.Int32(-1)), (A.shape[0], crdA_tmp[1])
                )
    for rest_v in range(tBprdB_res_k.shape[0], unroll_full=True):
        for n in range(tBprdB_res_k.shape[1], unroll_full=True):
            for k in range(tBprdB_res_k.shape[2], unroll_full=True):
                crdB_tmp = tBcrdB[(0, rest_v), n, k, 0]
                tBprdB_res_k[rest_v, n, k] = cute.elem_less(
                    (crdB_tmp[0], cutlass.Int32(-1)), (B.shape[0], crdB_tmp[1])
                )

    # ==========================================================================
    # Prologue
    # --------------------------------------------------------------------------
    # Prefetch GMEM2SMEM
    # Start async loads for prefetching the 0th k-residue tile
    smem_pipe_depth = cute.size(tAsA, mode=[3])
    num_tiles_k = cute.size(tAgA, mode=[3])
    gmem_pipe_read = cutlass.Int32(0)
    cute.copy(
        tiled_copy_a,
        tAgA[None, None, None, gmem_pipe_read],
        tAsA[None, None, None, 0],
        pred=tAprdA_res_k,
    )
    cute.copy(
        tiled_copy_b,
        tBgB[None, None, None, gmem_pipe_read],
        tBsB[None, None, None, 0],
        pred=tBprdB_res_k,
    )
    cute.arch.cp_async_commit_group()

    # Start async loads and fill the 1 to smem_pipe_deepth-1 pipes.
    gmem_pipe_read += 1
    if gmem_pipe_read >= num_tiles_k:
        gmem_pipe_read = cutlass.Int32(0)

    for pipe in range(1, smem_pipe_depth - 1, unroll_full=True):
        if pipe < num_tiles_k:
            cute.copy(
                tiled_copy_a,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, pipe],
                pred=tAprdA,
            )
            cute.copy(
                tiled_copy_b,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, pipe],
                pred=tBprdB,
            )
        cute.arch.cp_async_commit_group()

        gmem_pipe_read += 1
        if gmem_pipe_read >= num_tiles_k:
            gmem_pipe_read = cutlass.Int32(0)

    # If num_tiles_k is less than the smem depth,
    # clean the predicate tensor
    if num_tiles_k < smem_pipe_depth:
        for rest_v in range(tAprdA.shape[0], unroll_full=True):
            for m in range(tAprdA.shape[1], unroll_full=True):
                tAprdA[rest_v, m, 0] = cutlass.Boolean(0)
        for rest_v in range(tBprdB.shape[0], unroll_full=True):
            for n in range(tBprdB.shape[1], unroll_full=True):
                tBprdB[rest_v, n, 0] = cutlass.Boolean(0)

    # --------------------------------------------------------------------------
    # Prefetch SMEM2RMEM
    # Define A/B partitioning and C accumulators.
    # * tCsA (mma_atom, (permutation_m, num_mma_m), num_mma_k, smem_pipe_depth)
    # * tCsB (mma_atom, (permutation_n, num_mma_n), num_mma_k, smem_pipe_depth)
    # * tCgC (mma_atom, (permutation_m, num_mma_m), (permutation_n, num_mma_n))
    # * tCrA (mma_atom, (permutation_m, num_mma_m), num_mma_k)
    # * tCrB (mma_atom, (permutation_n, num_mma_n), num_mma_k)
    # * tCrC (mma_atom, (permutation_m, num_mma_m), (permutation_n, num_mma_n))
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    if VERBOSE:
        print(f"{LOG} tCsA {tCsA}")
        print(f"{LOG} tCsB {tCsB}")
        print(f"{LOG} tCgC {tCgC}")
        print(f"{LOG} tCrA {tCrA}")
        print(f"{LOG} tCrB {tCrB}")
        print(f"{LOG} tCrC {tCrC}")

    # Init SMEM: [R][#][#][W]
    #            [√][√][√][×]
    smem_pipe_read = cutlass.Int32(0)
    smem_pipe_write = cutlass.Int32(smem_pipe_depth - 1)

    tCsA_ = tCsA[None, None, None, smem_pipe_read]
    tCsB_ = tCsB[None, None, None, smem_pipe_read]

    num_mma_k = cute.size(tCrA, mode=[2])
    if num_mma_k > 1:
        # Wait until the first prefetched tile is loaded in
        cute.arch.cp_async_wait_group(smem_pipe_depth - 2)
        cute.arch.barrier()
        # Prefetch the first rmem from the first k-tile
        cute.autovec_copy(tCsA_[None, None, 0], tCrA[None, None, 0])
        cute.autovec_copy(tCsB_[None, None, 0], tCrB[None, None, 0])

    # ======================================================================
    # Mainloop.
    # `Doclink <cutlass/examples/python/CuTeDSL/ampere/sgemm.py#L523>`
    for _ in range(num_tiles_k, unroll_full=False):
        for mma_k_idx in range(num_mma_k, unroll_full=True):

            # If this mma tile finished. Insert a barrier to wait for the next tile cp.
            if mma_k_idx == num_mma_k - 1:
                tCsA_ = tCsA[None, None, None, smem_pipe_read]
                tCsB_ = tCsB[None, None, None, smem_pipe_read]
                cute.arch.cp_async_wait_group(smem_pipe_depth - 2)
                cute.arch.barrier()

            # Load next A, B mma frags smem2rmem
            mma_k_next = (mma_k_idx + 1) % num_mma_k
            cute.autovec_copy(tCsA_[None, None, mma_k_next], tCrA[None, None, mma_k_next])
            cute.autovec_copy(tCsB_[None, None, mma_k_next], tCrB[None, None, mma_k_next])

            # If start a new tile gemm, fetch the next A tile, then fetch the next B tile.
            # - In order to better interleave global memory access and compute instructions,
            #   the tile-fetchings are issued between mma ops.
            if mma_k_idx == 0:
                cute.copy(
                    tiled_copy_a,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, smem_pipe_write],
                    pred=tAprdA,
                )

            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, mma_k_idx],
                tCrB[None, None, mma_k_idx],
                tCrC,
            )

            if mma_k_idx == 0:
                cute.copy(
                    tiled_copy_b,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, smem_pipe_write],
                    pred=tBprdB,
                )

            # Commit and then update s/r-pipe ptrs.
            if mma_k_idx == 0:
                cute.arch.cp_async_commit_group()

                smem_pipe_write = smem_pipe_read
                smem_pipe_read += 1
                if smem_pipe_read >= smem_pipe_depth:
                    smem_pipe_read = cutlass.Int32(0)
                # - After copying all tiles, we avoid clearing the predicate tensor
                #   in the mainloop to prevent increasing its instruction count.
                #   Instead, we continue copying the first tile that won't be used.
                # - Note that the 0-th tile is skipped due to its irregular shape may cause
                #   illegal memory accesses.
                gmem_pipe_read += 1
                if gmem_pipe_read >= num_tiles_k:
                    gmem_pipe_read = cutlass.Int32(1)

    # ===================================================================
    # Epilogue
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()
    tCrC.store(epilogue_op(tCrC.load()))

    # Predicate the C's bounds
    crdC = cute.make_identity_tensor(gC.shape)
    tCcrdC = thr_mma.partition_C(crdC)
    tCpredC = cute.make_fragment(tCrC.layout, cutlass.Boolean)
    offset_m, offset_n = [C.shape[i] - cutlass.Int32(cta_tiler[i]) * tiler_coord[i] for i in (0, 1)]
    for i in range(cute.size(tCpredC.shape), unroll_full=True):
        tCpredC[i] = cute.elem_less(tCcrdC[i], (offset_m, offset_n))

    # crdC = cute.make_identity_tensor(C.shape)
    # tCcrdC = cute.local_tile(crdC, cta_tiler, tiler_coord, proj=(1, 1, None))
    # tCcrdC = thr_mma.partition_C(tCcrdC)
    # tCpredC = cute.make_fragment(tCcrdC.layout, cutlass.Boolean)
    # for i in range(cute.size(tCpredC.shape)):
    #      tCpredC[i] = cute.elem_less(tCcrdC[i], C.shape)

    c_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
    cute.copy(c_copy_atom, tCrC, tCgC, pred=tCpredC)

    return


@cute.jit
def sgemm(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
    copy_bits: cutlass.Constexpr = 128,
):
    # num_stages for overlapping loading and computation
    # Each cta handles a tile of size (tile_m, K)@A and (tile_n, K)@B produce (tile_m, tile_n)@C
    # Every time loads (tile_m, tile_k)@A and (tile_n, tile_k)@B, loop K with tile_k
    # Default cta threads 256，So the mma shape is (16, 16), each thread loads 4 elems A and 4 B.
    tile_m, tile_n, tile_k = cta_tiler
    mma_m, mma_n = mma_tiler
    threads = mma_m * mma_n

    # For GEMM TN, A is LayoutEnum.ROW_MAJOR, B is LayoutEnum.COL_MAJOR
    # For GEMM TT, A is LayoutEnum.ROW_MAJOR, B is LayoutEnum.ROW_MAJOR
    a_major_mode = utils.LayoutEnum.from_tensor(a)
    b_major_mode = utils.LayoutEnum.from_tensor(b)

    # Create layouts for shared memory for A and B
    # use padding to avoid bank conflict when ldg then sts.
    # TODO Why 4?
    padding_a = 4 if a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
    padding_b = 4 if b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
    smem_layout_a = cute.make_layout(
        (tile_m, tile_k, num_stages), stride=(1, tile_m + padding_a, tile_k * (tile_m + padding_a))
    )
    smem_layout_b = cute.make_layout(
        (tile_n, tile_k, num_stages), stride=(1, tile_n + padding_b, tile_k * (tile_n + padding_b))
    )
    smem_size = sum(
        [cute.size_in_bytes(cutlass.Float32, lo) for lo in [smem_layout_a, smem_layout_b]]
    )  # cute.cosize

    # Create copy partition.
    # - If A/B is ROW_MAJOR, i,e., A@T (M,K):(K,1), B@N (N,K):(K,1)
    #   don't need vl since multiple threads's LDGs can coalesce.
    # - If A/B is COL_MAJOR, i.e., A@N (M,K):(1,M), B@T (N,K):(1,N)
    #   try to use 128bit vector load.

    vl_bytes = copy_bits // 8
    vl_elems = copy_bits // 32

    if cutlass.const_expr(
        a_major_mode == utils.LayoutEnum.COL_MAJOR and a.layout.max_alignment % vl_bytes == 0
    ):
        thr_layout_a = cute.make_ordered_layout(
            (tile_m // vl_elems, threads * vl_elems // tile_m), order=(0, 1)
        )
        val_layout_a = cute.make_layout((vl_elems, 1))
        async_copy_atom_a = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.Float32,
            num_bits_per_copy=a.element_type.width * vl_elems,
        )
    else:
        thr_layout_a = cute.make_ordered_layout((threads // tile_k, tile_k), order=(1, 0))
        val_layout_a = cute.make_layout((1, 1))
        async_copy_atom_a = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.Float32,
            num_bits_per_copy=a.element_type.width,
        )

    if cutlass.const_expr(
        b_major_mode == utils.LayoutEnum.COL_MAJOR and b.layout.max_alignment % vl_bytes == 0
    ):
        thr_layout_b = cute.make_ordered_layout(
            (tile_n // vl_elems, threads * vl_elems // tile_n), order=(0, 1)
        )
        val_layout_b = cute.make_layout((vl_elems, 1))
        async_copy_atom_b = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.Float32,
            num_bits_per_copy=b.element_type.width * vl_elems,
        )
    else:
        thr_layout_b = cute.make_ordered_layout((threads // tile_k, tile_k), order=(1, 0))
        val_layout_b = cute.make_layout((1, 1))
        async_copy_atom_b = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.Float32,
            num_bits_per_copy=b.element_type.width,
        )

    tiled_copy_a = cute.make_tiled_copy_tv(async_copy_atom_a, thr_layout_a, val_layout_a)
    tiled_copy_b = cute.make_tiled_copy_tv(async_copy_atom_b, thr_layout_b, val_layout_b)

    # Create layouts for GEMM
    # - The MmaUniversalOp has a trivial 1x1x1 MMA trait.
    # - The permutation = vl_elems(4) means each thread loads 4 contiguous A/B elems
    #   from smem to regs, i.e., lds.128.
    # - mma-thread-tile illustration:
    #   https://developer-blogs.nvidia.com/wp-content/uploads/2017/12/fig-06-warp-tile-structure.png
    mma_op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    mma_atoms_layout = cute.make_layout((mma_m, mma_n, 1), stride=(mma_n, 1, 0))
    permutation_m = cute.make_ordered_layout((mma_m, vl_elems), order=(1, 0))
    permutation_n = cute.make_ordered_layout((mma_n, vl_elems), order=(1, 0))
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        atom_layout_mnk=mma_atoms_layout,
        permutation_mnk=(permutation_m, permutation_n, None),
    )

    grid_dim = [*cute.ceil_div(c.shape, (tile_m, tile_n)), 1]
    block_dim = [threads, 1, 1]

    if VERBOSE:
        print(f"{LOG} Tensor A {a.type}")
        print(f"{LOG} Tensor B {b.type}")
        print(f"{LOG} Tensor C {c.type}")
        print(f"{LOG} Gemm tile size {cta_tiler}")
        print(f"{LOG} Smem layout A {smem_layout_a}")
        print(f"{LOG} Smem layout B {smem_layout_b}")
        print(f"{LOG} Copy layout A {tiled_copy_a}")
        print(f"{LOG} Copy layout B {tiled_copy_b}")
        print(f"{LOG} Mma layout {tiled_mma}")
        print(f"{LOG} Sgemm grid {grid_dim}")
        print(f"{LOG} Sgemm block {block_dim}")

    sgemm_kernel(
        a,
        b,
        c,
        smem_layout_a,
        smem_layout_b,
        tiled_copy_a,
        tiled_copy_b,
        tiled_mma,
        epilogue_op,
    ).launch(grid=grid_dim, block=block_dim, smem=smem_size, stream=stream)


def run_sgemm(
    M: int,
    N: int,
    K: int,
    blas: str = "tn",
    skip_verify: bool = False,
    dynamic_layout: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 100,
):
    print("Running sgemm with M={}, N={}, K={}, BLAS T{}.".format(M, N, K, blas.upper()))

    ta, tb = blas
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    def tensor_generator(return_torch_tensor: bool = False):

        if ta == "n":
            a = numpy.random.uniform(0, 100, size=(K, M)).transpose()  # (M, K):(1, M)
        else:
            a = numpy.random.uniform(0, 100, size=(M, K))  # (M, K):(K, 1)

        if tb == "n":
            b = numpy.random.uniform(0, 100, size=(N, K))  # (N, K):(K, 1)
        else:
            b = numpy.random.uniform(0, 100, size=(K, N)).transpose()  # (N, K):(1, N)

        a = torch.from_numpy(a).to(device=torch.device("cuda"), dtype=torch.float32)
        b = torch.from_numpy(b).to(device=torch.device("cuda"), dtype=torch.float32)
        c = torch.zeros((M, N), device=torch.device("cuda"), dtype=torch.float32)

        a_tensor = from_dlpack(a, assumed_align=16)
        b_tensor = from_dlpack(b, assumed_align=16)
        c_tensor = from_dlpack(c, assumed_align=16)

        if dynamic_layout:
            a_tensor = a_tensor.mark_layout_dynamic(leading_dim=1 if ta == "t" else 0)
            b_tensor = b_tensor.mark_layout_dynamic(leading_dim=1 if tb == "n" else 0)
            c_tensor = c_tensor.mark_layout_dynamic(leading_dim=1)

        if return_torch_tensor:
            return a, b, c, a_tensor, b_tensor, c_tensor
        return a_tensor, b_tensor, c_tensor

    workspace_generator = lambda: testing.JitArguments(*tensor_generator(), current_stream)

    _a, _b, _c, _a_tensor, _b_tensor, _c_tensor = tensor_generator(True)

    # Compile
    compile_tic = time.perf_counter()
    sgemm_func = cute.compile(sgemm, _a_tensor, _b_tensor, _c_tensor, current_stream)
    print(f"Kernel compiled in {time.perf_counter() - compile_tic:.4f} seconds")

    # Verify
    if not skip_verify:
        sgemm_func(_a_tensor, _b_tensor, _c_tensor, current_stream)
        torch.cuda.synchronize()
        _c_ref = torch.einsum("mk,nk->mn", _a, _b)
        torch.testing.assert_close(_c.cpu(), _c_ref.cpu(), atol=1e-03, rtol=1e-05)
        print("Verification passed!")
    else:
        print("Verification skipped...")

    # Benchmarking
    # Use workspaces to avoid L2 cache is not cold.

    # Bench torch
    torch.cuda.empty_cache()
    for _ in range(warmup_iterations):
        _ = torch.einsum("mk,nk->mn", _a, _b)
    torch.cuda.synchronize()
    torch_tic = time.perf_counter()
    for _ in range(iterations):
        _ = torch.einsum("mk,nk->mn", _a, _b)
    torch.cuda.synchronize()
    torch_avg_time_us = (time.perf_counter() - torch_tic) * 1e6 / iterations
    print(f"Torch kernel execution time: {torch_avg_time_us / 1e3:.2f} ms")
    print(f"Torch achieved TOPS: {(2 * M * N * K) / torch_avg_time_us / 1e6:.2f}")

    # Bench CuTe
    torch.cuda.empty_cache()
    workspace_bytes = (M * K + N * K + M * N) * 4
    workspace_count = testing.get_workspace_count(workspace_bytes, warmup_iterations, iterations)

    avg_time_us = testing.benchmark(
        sgemm_func,
        workspace_generator=workspace_generator,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        stream=current_stream,
        use_cuda_graphs=False,
    )
    print(f"Cute kernel execution time: {avg_time_us / 1e3:.2f} ms")
    print(f"Cute achieved TOPS: {(2 * M * N * K) / avg_time_us / 1e6:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise ops to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", "-M", default=4096, type=int)
    parser.add_argument("--N", "-N", default=2048, type=int)
    parser.add_argument("--K", "-K", default=1024, type=int)
    parser.add_argument("--warmup-iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--blas", type=str, default="tn", choices=["tn", "tt", "nn", "nt"])
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--dynamic-layout", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    check_cuda()

    VERBOSE = args.verbose

    run_sgemm(
        args.M,
        args.N,
        args.K,
        blas=args.blas,
        skip_verify=args.skip_verify,
        dynamic_layout=args.dynamic_layout,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    print("PASS!")
