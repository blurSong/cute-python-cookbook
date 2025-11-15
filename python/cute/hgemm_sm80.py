"""
A dense HGEMM using the sm80 tensor cores in CUTE DSL.

The HGEMM implementation handles only row-major and col-major layouts.
To bridge the gap of GEMM order between BLAS and CUTE, we can use the following definitions:
------------------------------------------
Blas      T                   N
------------------------------------------
A         (M, K):(K, 1)      (M, K):(1, M)
B         (N, K):(1, N)      (N, K):(K, 1)
------------------------------------------
See also:
    https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.html#aside-m-major-n-major-k-major

Key optimizations:
    TODO.

References:
    TODO.

To run:
    TODO.

Constraints:
    TODO.
"""

import time
import math
import torch
import argparse
from typing import Type, List, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

from utils import check_cuda, benchmark_torch

VERBOSE = False
LOG = "[CuTe Info]"


# GLOBAL =================================================
# DATA TYPES ---------------------------------------------
gemm_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
# HGEMM CONFIGURATIONS -----------------------------------
cta_tiler = (128, 128, 32)
mma_inst_shape = (16, 8, 16)
mma_atom_shape = (2, 2, 1)
threads = 128
num_stages = 3
copy_bits = 128
bytes_alignment = 16
# PARAMETERS DERIVED -------------------------------------
vl = copy_bits // gemm_dtype.width
tile_m, tile_n, tile_k = cta_tiler
mma_inst_m, mma_inst_n, mma_inst_k = mma_inst_shape
mma_atom_m, mma_atom_n, mma_atom_k = mma_atom_shape
# ASSERTIONS ---------------------------------------------
assert tile_m % (mma_atom_m * mma_inst_m) == 0
assert tile_n % (mma_atom_n * mma_inst_n) == 0
assert mma_atom_k == 1
assert tile_k % mma_inst_k == 0
assert bytes_alignment
# ========================================================


def raster_tile(i, j, f):
    new_i = i // f
    new_j = (i % f) + (j * f)
    return (new_i, new_j)


def make_smem_layout_AB(
    major_mode: LayoutEnum,
    smem_tiler: Tuple[int, int, int],
):
    """Make shared memory layout for A and B tiles with swizzling.

    The PTX `ldmatrix` section describes the ldmatrix instruction in detail.
    The PTX `Shared Memory Layout and Swizzling` section provides all useful `swizzle layout atom`s.

    Since we are doing fp16 gemm, we manually apply a `<3,3,3>` swizzle onto a 8x128B tile.
    I.e., the `128B Swizzling with 16B atomicity` follows the PTX naming convention.

    Refer to this awesome blog for the ldmatrix/swizzle mechanism:
        - https://yang-yifan.github.io/blogs/mma_swizzle/mma_swizzle.html

    Params:
        - major_mode: `LayoutEnum.ROW_MAJOR` or `LayoutEnum.COL_MAJOR`
        - smem_tiler: smem sizes of `row`, `col` and `depth`
    """
    contiguous_size = smem_tiler[1] if major_mode == LayoutEnum.ROW_MAJOR else smem_tiler[0]
    contiguous_size = min(64, contiguous_size)
    # swizzle layout atom is upto `128B Swizzling with 16B atomicity`

    m_base = 3
    # one row of 2^3=8 values loaded by 1 thread quad
    s_shift = 3
    # 8 rows per ldmatrix.m8n8.b16
    b_bits = min(int(math.log2(contiguous_size / vl)), 3)
    # - b_bits is limited upto 3 because smem bank size is 32x4B.
    #   For ldmatrix.m8n8.b16 the up-most swizzle BBits is log2(128B/16B)=3.
    # - By default vl is 8 values (16B), making the BBits exactly 3.
    #   If vl is larger, to avoid smem bank conflicts,
    #   we will need less BBits to cover the contiguous_size.
    swizzle = cute.make_swizzle(b_bits, m_base, s_shift)

    if major_mode == LayoutEnum.ROW_MAJOR:
        # K-Major swizzle
        layout_atom_outer = cute.make_ordered_layout((8, contiguous_size), order=(1, 0))
    else:
        # MN-Major swizzle
        layout_atom_outer = cute.make_ordered_layout((contiguous_size, 8), order=(0, 1))

    layout_atom = cute.make_composed_layout(swizzle, 0, layout_atom_outer)
    layout = cute.tile_to_shape(layout_atom, smem_tiler, order=(0, 1, 2))
    return layout


def make_smem_layout_C(
    smem_tiler: Tuple[int, int],
):
    """C is always row-major layout (K-Major SMEM atom).

    Params:
        - smem_tiler: smem sizes of `row` and `col`
    """
    contiguous_size = smem_tiler[1]
    m_base = 3
    s_shift = 4
    b_bits = min(int(math.log2(contiguous_size / vl)), 3)
    swizzle = cute.make_swizzle(b_bits, m_base, s_shift)

    layout_atom_outer = cute.make_ordered_layout((8, contiguous_size), order=(1, 0))
    layout_atom = cute.make_composed_layout(swizzle, 0, layout_atom_outer)
    layout = cute.tile_to_shape(layout_atom, smem_tiler, order=(0, 1))
    return layout


def make_tiled_copy_ABC(
    copy_atom: cute.CopyAtom,
    major_mode: LayoutEnum,
    cta_tiler: Tuple[int, int],
):
    """Make GEMM tiled copy A and B for GMEM to SMEM.

    The thread layout follows the major_mode of the GMEM. Each thread copies vl values per access.

    Params:
        - cta_tiler: cta tile sizes. A: `(tile_m, tile_k)`, B: `(tile_n, tile_k)`, C: `(tile_m, tile_n)`
    """
    if major_mode == LayoutEnum.ROW_MAJOR:
        thr_tiler_1 = cta_tiler[1] // vl
        thr_tiler_0 = threads // thr_tiler_1
        order = (1, 0)
    else:
        thr_tiler_0 = cta_tiler[0] // vl
        thr_tiler_1 = threads // thr_tiler_0
        order = (0, 1)
    thr_layout = cute.make_ordered_layout((thr_tiler_0, thr_tiler_1), order=order)

    if major_mode == LayoutEnum.ROW_MAJOR:
        val_tiler = (1, vl)
    else:
        val_tiler = (vl, 1)
    val_layout = cute.make_layout(val_tiler)

    tiled_copy_tv = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    return tiled_copy_tv


@cute.kernel
def hgemm_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    smem_layout_A: cute.Layout | cute.ComposedLayout,
    smem_layout_B: cute.Layout | cute.ComposedLayout,
    smem_layout_C: cute.Layout | cute.ComposedLayout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_copy_C: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    raster_factor: cutlass.Int32,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    tidx = cute.arch.thread_idx()[0]
    bidx, bidy, bidz = cute.arch.block_idx()

    offset_bidx, offset_bidy = raster_tile(bidx, bidy, raster_factor)
    original_grid_dim = cute.ceil_div(C.shape, (tile_m, tile_n, 1))
    if offset_bidx >= original_grid_dim[0] or offset_bidy >= original_grid_dim[1]:
        pass

    cta_coord = (offset_bidx, offset_bidy, None)

    # Get the appropriate tiles for this thread block.
    # * gA: (tile_m, tile_k, k_tiles)
    # * gB: (tile_n, tile_k, k_tiles)
    # * gC: (tile_m, tile_n)
    # - The`proj` is applied to both cta_tiler and the cta_coord
    # - Using local_tile is equal to use tiling & indexing:
    #   gA = zipped_divide(A[batch_coord], (tile_m, tile_k))[(offset_bidx, None)]
    coord_batch = (None, None, bidz)
    gA = cute.local_tile(A[coord_batch], cta_tiler, cta_coord, proj=(1, None, 1))
    gB = cute.local_tile(B[coord_batch], cta_tiler, cta_coord, proj=(None, 1, 1))
    gC = cute.local_tile(C[coord_batch], cta_tiler, cta_coord, proj=(1, 1, None))

    # Shifting the pointer of gA and gB to the left along the k demesion.
    # Making the first tile irregular for prefethching.
    # Thus avoid checking boundray condition whithin the mma main loop.

    # residual_k is negative
    residual_k = cute.size(A, mode=[1]) - cutlass.Int32(tile_k) * cute.size(gA, mode=[2])
    gA = cute.domain_offset((0, residual_k, 0), gA)
    gB = cute.domain_offset((0, residual_k, 0), gB)
    # Note that re-making gA gB and gC 16B aligned is very non-universal
    gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
    gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
    gC = cute.make_tensor(gC.iterator.align(16), gC.layout)

    # Making coord tensors, appling the same tiling/shifting for I/O predication
    crdA = cute.make_identity_tensor(A.shape)
    crdB = cute.make_identity_tensor(B.shape)
    crdC = cute.make_identity_tensor(C.shape)
    crdA = cute.local_tile(crdA[coord_batch], cta_tiler, cta_coord, proj=(1, None, 1))
    crdB = cute.local_tile(crdB[coord_batch], cta_tiler, cta_coord, proj=(None, 1, 1))
    crdC = cute.local_tile(crdC[coord_batch], cta_tiler, cta_coord, proj=(1, 1, None))
    crdA = cute.domain_offset((0, residual_k, 0), crdA)
    crdB = cute.domain_offset((0, residual_k, 0), crdB)

    # Allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(gemm_dtype, smem_layout_A, byte_alignment=16)
    sB = smem.allocate_tensor(gemm_dtype, smem_layout_B, byte_alignment=16)
    sC = smem.allocate_tensor(gemm_dtype, smem_layout_C, byte_alignment=16)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)
    # * tAgA ((atom_v, rest_v), copy_m, copy_k, k_tiles)
    # * tAsA ((atom_v, rest_v), copy_m, copy_k, num_stages)
    # * tBgB ((atom_v, rest_v), copy_n, copy_k, k_tiles)
    # * tBsB ((atom_v, rest_v), copy_n, copy_k, num_stages)
    # * tCgC ((atom_v, rest_v), copy_m, copy_n)
    # * tCsC ((atom_v, rest_v), copy_m, copy_n)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)
    # - Note that the S/D of C is different
    # - Note that the _epilogue is the tiled_copy view of smem C and
    #   used in the epilogue SMEM->GMEM.
    tCgC_epilogue = thr_copy_C.partition_D(gC)
    tCsC_epilogue = thr_copy_C.partition_S(sC)

    tAcrdA = thr_copy_A.partition_S(crdA)
    tBcrdB = thr_copy_B.partition_S(crdB)
    tCcrdC = thr_copy_C.partition_S(crdC)

    if VERBOSE:
        print(f"{LOG} gA {gA}")
        print(f"{LOG} gB {gB}")
        print(f"{LOG} gC {gC}")
        print(f"{LOG} sA {sA}")
        print(f"{LOG} sB {sB}")
        print(f"{LOG} sC {sC}")
        print(f"{LOG} tAgA {tAgA}")
        print(f"{LOG} tAsA {tAsA}")
        print(f"{LOG} tBgB {tBgB}")
        print(f"{LOG} tBsB {tBsB}")
        print(f"{LOG} tCgC epilogue {tCgC_epilogue}")
        print(f"{LOG} tCsC epilogue {tCsC_epilogue}")

    # Making predicators of A, B and C
    # - To avoid creating multiple predicators for normal/residual blocks
    #   For AB's MN demension, predication are stored in predicators
    #   For AB's K demension, predication is handled via if/else
    # - Note that tXpreX's 0th dim is rest_v in (atom_v, rest_v). This is due to
    #   that the predicators is checked at the granularity of the Copy Atom.
    # * tAprdA (rest_v, copy_m, copy_k)
    # * tBprdB (rest_v, copy_n, copy_k)
    # * tCprdC (rest_v, copy_m, copy_n)
    tAprdA = cute.make_rmem_tensor(
        cute.make_layout(
            (tAgA.shape[0][1], tAgA.shape[1], tAgA.shape[2]),
            stride=(tAgA.shape[1], 1, 0),
        ),
        dtype=cutlass.Boolean,
    )
    tBprdB = cute.make_rmem_tensor(
        cute.make_layout(
            (tBgB.shape[0][1], tBgB.shape[1], tBgB.shape[2]),
            stride=(tBgB.shape[1], 1, 0),
        ),
        dtype=cutlass.Boolean,
    )
    tCprdC = cute.make_rmem_tensor(
        cute.make_ordered_layout(
            (tCgC_epilogue.shape[0][1], tCgC_epilogue.shape[1], tCgC_epilogue.shape[2]),
            order=(2, 1, 0),
        ),
        dtype=cutlass.Boolean,
    )

    for i in range(tAprdA.shape[0]):
        for j in range(tAprdA.shape[1]):
            coord_tmp = ((0, i), j, 0, 0)
            tAprdA[i, j, 0] = cute.elem_less(tAcrdA[coord_tmp][0], A.shape[0])
    for i in range(tBprdB.shape[0]):
        for j in range(tBprdB.shape[1]):
            coord_tmp = ((0, i), j, 0, 0)
            tBprdB[i, j, 0] = cute.elem_less(tBcrdB[coord_tmp][1], B.shape[0])
    for i in range(tCprdC.shape[0]):
        for j in range(tCprdC.shape[1]):
            for k in range(tCprdC.shape[2]):
                coord_tmp = ((0, i), j, k)
                tCprdC[i, j, k] = cute.elem_less(tCcrdC[coord_tmp], C.shape)

    # =============================== Prefetch Prologue ===============================
    # ---------------------------------------------------------------------------------
    # - Note that here fill the smem buffers to zero at the beginning.
    #   The trick is, the first tile of A, B may be irregular, but the OOB
    #   values will be 0 in smem. Making the SMEM->RMEM harmless.
    tAsA.fill(0)
    tBsB.fill(0)
    cute.arch.sync_threads()

    # Prefetch the first tile GMEM->SMEM
    # - Since the domain_offset is applied, now the first tile of A, B may be irregular.
    #   We reuse the predicators A.B to handle the M/N boundary conditions.
    # - Before that, we handle the k boundary one-by-one using if-else statement.
    #   If one k_index is OOB, its corresponding coord tensore will be NEGATIVE.
    for k in range(tAprdA.shape[2]):
        coord_g2s = (None, None, k, 0)
        if cute.elem_less(-1, tAcrdA[0, 0, k, 0][1]):
            cute.copy(tiled_copy_A, tAgA[coord_g2s], tAsA[coord_g2s], pred=tAprdA[coord_g2s[:-1]])
    for k in range(tBprdB.shape[2]):
        coord_g2s = (None, None, k, 0)
        if cute.elem_less(-1, tBcrdB[0, 0, k, 0][1]):
            cute.copy(tiled_copy_B, tBgB[coord_g2s], tBsB[coord_g2s], pred=tBprdB[coord_g2s[:-1]])
    cute.arch.cp_async_commit_group()

    # Fetching the remaining num_stages-1-1 tiles GMEM->SMEM
    num_k_tiles = cute.size(tAgA, mode=[3])
    k_tile_index_gmem = cutlass.Int32(1)
    k_tile_index_smem = cutlass.Int32(1)
    for _ in range(1, num_stages - 1):
        # If reach num_k_tiles, clear the predictors
        if k_tile_index_smem == num_k_tiles:
            tAprdA.fill(0)
            tBprdB.fill(0)
        coord_gmem = (None, None, None, k_tile_index_gmem)
        coord_smem = (None, None, None, k_tile_index_smem)
        cute.copy(tiled_copy_A, tAgA[coord_gmem], tAsA[coord_smem], pred=tAprdA)
        cute.copy(tiled_copy_B, tBgB[coord_gmem], tBsB[coord_smem], pred=tBprdB)
        k_tile_index_gmem += 1
        k_tile_index_smem += 1
        cute.arch.cp_async_commit_group()

    # Tile MMA thread partitions
    # - Note that the tCsA tCsB are thr_mma views of smem A/B and
    #   only used to create the tCrA, tCrB fragments.
    # - Note that the tCsC_mma is the thr_mma view of smem C and
    #   used in the mma loop and epilogue.
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCsC_mma = thr_mma.partition_C(sC)
    tCgC_mma = thr_mma.partition_C(gC)
    tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = thr_mma.make_fragment_C(tCgC_mma)
    tCrC.fill(0.0)

    # Copy Atom A/B retiling
    # Creating the ldmatrix copy atom with {.trans} {.num}
    ldmatrix = cute.nvgpu.warp.LdMatrix8x8x16bOp
    trans_A = LayoutEnum.from_tensor(A) != LayoutEnum.ROW_MAJOR
    trans_B = LayoutEnum.from_tensor(B) != LayoutEnum.ROW_MAJOR
    copy_atom_s2r_A = cute.make_copy_atom(ldmatrix(trans_A, 4), gemm_dtype)
    copy_atom_s2r_B = cute.make_copy_atom(ldmatrix(trans_B, 4), gemm_dtype)

    # Creating the tiled copy SMEM->RMEM
    # - The key point is to matche the tv layout of the tiled mma.
    # - make_tiled_copy_A is just a wapper of make_tiled_copy_tv with
    #   tiled_mma's A's tv_layout.
    tiled_copy_s2r_A = cute.make_tiled_copy_A(copy_atom_s2r_A, tiled_mma)
    tiled_copy_s2r_B = cute.make_tiled_copy_B(copy_atom_s2r_B, tiled_mma)

    thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
    thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)

    tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
    tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
    tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
    tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

    if VERBOSE:
        print(f"{LOG} tCsA {tCsA}")
        print(f"{LOG} tCsB {tCsB}")
        print(f"{LOG} tCsC mma {tCsC_mma}")
        print(f"{LOG} tCgC mma {tCgC_mma}")
        print(f"{LOG} tCrA {tCrA}")
        print(f"{LOG} tCrB {tCrB}")
        print(f"{LOG} tCrC {tCrC}")
        print(f"{LOG} tCsA copy_view {tCsA_copy_view}")
        print(f"{LOG} tCsB copy_view {tCsB_copy_view}")
        print(f"{LOG} tCrA copy_view {tCrA_copy_view}")
        print(f"{LOG} tCrB copy_view {tCrB_copy_view}")

    # Prefetch SMEM->RMEM for the first MMA tile
    smem_pipe_read = cutlass.Int32(0)
    smem_pipe_write = cutlass.Int32(num_stages - 1)

    tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
    tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

    num_k_frags = cute.size(tCrA, mode=[2])
    if num_k_frags > 1:
        # SMEM has #num_stages tiles, #num_stages-1 are in the prefetching pipeline
        # cp_async_wait_group waits for the earliest tile to be ready
        cute.arch.cp_async_wait_group(num_stages - 2)
        cute.arch.sync_threads()
        # Then, prefetching the first mma frag tile SMEM->RMEM
        coord_frag = (None, None, 0)
        cute.copy(tiled_copy_s2r_A, tCsA_p[coord_frag], tCrA_copy_view[coord_frag])
        cute.copy(tiled_copy_s2r_B, tCsB_p[coord_frag], tCrB_copy_view[coord_frag])

    # ================================ Main Loop ================================
    # ---------------------------------------------------------------------------
    # 1.Shared memory pipeline (gmem -> smem):
    #   The default smem pipeline depth is 3, meaning that for shared
    #   memory buffers, we allocate three times the size described by the
    #   CTA tiler. We prefetch 2 of these buffers before entering the main
    #   loop. Considering only the transfer from global memory to shared
    #   memory, the general structure of the mainloop is:
    #       (1) copy k-tile from gmem to smem;
    #       (2) perform gemm computation on k-tile;
    #       (3) wait for the next copy to finish.
    #   The cute.arch.cp_async_wait_group(num_smem_stages - 2) command
    #   waits for the number of unfinished copies to be <= 1. The advantage
    #   of this approach is that it allows for simultaneous production
    #   (i.e., step (1)) and consumption (i.e., step (2)) of smem.
    #   A common misconception is to prefetch N buffers and rewrite
    #   the pipeline logic to wait on N-1 pending copies. The disadvantage
    #   of this approach is that it requires fully consuming a buffer in
    #   order to open an empty buffer for the next copy.
    # 2.Register pipeline (smem -> register):
    #   Similarly, the register pipeline produces i+1, consumes i, and
    #   produces i+2... Notably, i and i+1 do not use the same register,
    #   eliminating dependencies on the same register for better parallelism.
    # 3.Combining the smem and register pipelines results in the mainloop.

    for k_tile_index in range(num_k_tiles):
        # Fetching next k-tile GMEM->SMEM
        # - Note that current SMEM state is
        #   [F][O][O][E] F(FETCHED), O(ON-FLY), E(EMPTY)
        # - The initialized smem_pipe_r/w index are 0 and num_stages-1.
        #   Firstly the [E] smem tile will be fetched.
        if k_tile_index + num_stages - 1 < num_k_tiles:
            coord_gmem = (None, None, None, k_tile_index_gmem)
            coord_smem = (None, None, None, smem_pipe_write)
            cute.copy(tiled_copy_A, tAgA[coord_gmem], tAsA[coord_smem], pred=tAprdA)
            cute.copy(tiled_copy_B, tBgB[coord_gmem], tBsB[coord_smem], pred=tBprdB)
            cute.arch.cp_async_commit_group()
            # Update meta pointers of gmem/smem pipes
            k_tile_index_gmem += 1
            smem_pipe_write = smem_pipe_read
            smem_pipe_read = (smem_pipe_read + 1) % num_stages
        for k_frag_index in cutlass.range(num_k_frags, unroll_full=True):
            # - If the inner loop reaches the last fragment, we need to
            #   update tCsA_p/tCsB_p to prefetch the first fragment in the next smem.
            # - Note that the smem_pipe_read is already updated in the outer loop.
            if k_frag_index == num_k_frags - 1:
                coord_smem_next = (None, None, None, smem_pipe_read)
                tCsA_p = tCsA_copy_view[coord_smem_next]
                tCsB_p = tCsB_copy_view[coord_smem_next]
                cute.arch.cp_async_wait_group(num_stages - 2)
                cute.arch.sync_threads()
            # Then fetch the next mma frag SMEM->RMEM
            coord_frag_next = (None, None, (k_frag_index + 1) % num_k_frags)
            cute.copy(tiled_copy_s2r_A, tCsA_p[coord_frag_next], tCrA_copy_view[coord_frag_next])
            cute.copy(tiled_copy_s2r_B, tCsB_p[coord_frag_next], tCrB_copy_view[coord_frag_next])
            # Finally, perform mma on current frag
            # - Note that the tCrA/tCrB are in the mma view
            coord_frag_mma = (None, None, k_frag_index)
            cute.gemm(tiled_mma, tCrC, tCrA[coord_frag_mma], tCrB[coord_frag_mma], tCrC)

    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # =============================== Epilogue with fusion ===============================
    # ------------------------------------------------------------------------------------
    tCrD = cute.make_rmem_tensor_like(tCrC, dtype=gemm_dtype)
    tCrD.store(epilogue_op(tCrC.load()).to(gemm_dtype))

    # Note the difference between tCsC_epilogue and tCsC_mma
    cute.autovec_copy(tCrD, tCsC_mma)
    cute.arch.sync_threads()
    tCrC_epilogue = cute.make_rmem_tensor_like(tCsC_epilogue)
    cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

    # RMEM->GMEM C
    cute.copy(tiled_copy_C, tCrC_epilogue, tCgC_epilogue, pred=tCprdC)

    return


@cute.jit
def hgemm(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    # A is ROW_MAJOR@T and COL_MAJOR@N
    # B is COL_MAJOR@T and ROW_MAJOR@N
    major_mode_A = LayoutEnum.from_tensor(A)
    major_mode_B = LayoutEnum.from_tensor(B)
    major_mode_C = LayoutEnum.from_tensor(C)

    # Make SMEM layouts
    smem_layout_A = make_smem_layout_AB(major_mode_A, (tile_m, tile_k, num_stages))
    smem_layout_B = make_smem_layout_AB(major_mode_B, (tile_n, tile_k, num_stages))
    smem_layout_C = make_smem_layout_C((tile_m, tile_n))
    smem_size = sum(
        [
            cute.size_in_bytes(gemm_dtype, smem_layout)
            for smem_layout in [smem_layout_A, smem_layout_B, smem_layout_C]
        ]
    )

    # Make tiled copies
    # The asynchronous copy atom is for A, B GMEM->SMEM
    # The synchronous copy atom is for C SMEM->GMEM
    copy_atom_async = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
        ),  # LoadCacheMode.GLOBAL will bypass L1
        gemm_dtype,
        num_bits_per_copy=copy_bits,
    )
    copy_atom_sync = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gemm_dtype,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_A = make_tiled_copy_ABC(copy_atom_async, major_mode_A, (tile_m, tile_k))
    tiled_copy_B = make_tiled_copy_ABC(copy_atom_async, major_mode_B, (tile_n, tile_k))
    tiled_copy_C = make_tiled_copy_ABC(copy_atom_sync, major_mode_C, (tile_m, tile_n))

    # Make tiled MMA
    # hmma shape_mnk should be (16, 8, 8), (16, 8, 16)
    # mma_atom_layout defines how the MMA atom duplicates across the M, N and K
    # permutation_mnk defines the tiling of M, N and K values based on MMA shape and atom shape
    # * About how permutation_mnk works: https://github.com/NVIDIA/cutlass/discussions/1345
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(gemm_dtype, acc_dtype, shape_mnk=mma_inst_shape)
    mma_atom_layout = cute.make_layout(mma_atom_shape)
    permutation_mnk = (
        mma_atom_m * mma_inst_m,
        mma_atom_n * mma_inst_n * 2,  # hard coding to make M/N's mma balance
        mma_atom_k * mma_inst_k,
    )
    tiled_mma = cute.make_tiled_mma(mma_op, mma_atom_layout, permutation_mnk)

    # Make kernel launch parameters
    grid_dim = cute.ceil_div(C.shape, (tile_m, tile_n, 1))
    block_dim = (threads, 1, 1)

    # Optional optimization: CTA swizzling
    raster_factor = (
        8 if grid_dim[1] > 5 else (4 if grid_dim[1] > 2 else (2 if grid_dim[1] > 1 else 1))
    )
    grid_dim_rasterized = (
        grid_dim[0] * raster_factor,
        (grid_dim[1] + raster_factor - 1) // raster_factor,
        grid_dim[2],
    )

    if VERBOSE:
        print(f"{LOG} A {A}")
        print(f"{LOG} B {B}")
        print(f"{LOG} C {C}")
        print(f"{LOG} grid_dim {grid_dim}")
        print(f"{LOG} rasterized grid_dim {grid_dim_rasterized}")
        print(f"{LOG} block_dim {block_dim}")
        print(f"{LOG} smem_layout_A {smem_layout_A}")
        print(f"{LOG} smem_layout_B {smem_layout_B}")
        print(f"{LOG} smem_layout_C {smem_layout_C}")
        print(f"{LOG} tiled_copy_A {tiled_copy_A}")
        print(f"{LOG} tiled_copy_B {tiled_copy_B}")
        print(f"{LOG} tiled_copy_C {tiled_copy_C}")
        print(f"{LOG} tiled_mma {tiled_mma}")

    hgemm_kernel(
        A,
        B,
        C,
        smem_layout_A,
        smem_layout_B,
        smem_layout_C,
        tiled_copy_A,
        tiled_copy_B,
        tiled_copy_C,
        tiled_mma,
        raster_factor,
        epilogue_op,
    ).launch(
        grid=grid_dim_rasterized,
        block=block_dim,
        smem=smem_size,
    )


def run_hgemm(
    M: int,
    N: int,
    K: int,
    L: int = 1,
    blas: str = "tn",
    skip_verify: bool = False,
    dynamic_layout: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 100,
):
    print("Running hgemm with M={}, N={}, K={}, L={}, BLAS {}.".format(M, N, K, L, blas.upper()))

    a_transpose, b_transpose = [t == "t" for t in blas]

    def tensor_generator(return_type: str = "all"):
        assert return_type in ["all", "cute_only", "torch_only"]

        if a_transpose:
            a = torch.empty((L, M, K)).permute(1, 2, 0)  # (M, K, L):(K, 1, MK)
        else:
            a = torch.empty((L, K, M)).permute(2, 1, 0)  # (M, K, L):(1, M, MK)

        if b_transpose:
            b = torch.empty((L, K, N)).permute(2, 1, 0)  # (N, K, L):(1, N, NK)
        else:
            b = torch.empty((L, N, K)).permute(1, 2, 0)  # (N, K, L):(K, 1, NK)

        # C is always row-major
        c = torch.zeros((L, M, N)).permute(1, 2, 0)  # (M, N, L):(N, 1, MN)

        a = a.random_(-114, 514).to(device=torch.device("cuda"), dtype=torch.float16)
        b = b.random_(-114, 514).to(device=torch.device("cuda"), dtype=torch.float16)
        c = c.to(device=torch.device("cuda"), dtype=torch.float16)

        if return_type == "torch_only":
            return a, b, c

        a_tensor = from_dlpack(a, assumed_align=bytes_alignment)
        b_tensor = from_dlpack(b, assumed_align=bytes_alignment)
        c_tensor = from_dlpack(c, assumed_align=bytes_alignment)

        if dynamic_layout:
            a_tensor = a_tensor.mark_layout_dynamic(leading_dim=1 if a_transpose else 0)
            b_tensor = b_tensor.mark_layout_dynamic(leading_dim=0 if b_transpose else 1)
            c_tensor = c_tensor.mark_layout_dynamic(leading_dim=1)

        if return_type == "cute_only":
            return a_tensor, b_tensor, c_tensor

        return (a, a_tensor), (b, b_tensor), (c, c_tensor)

    # verification and compilation
    (a_torch, a_tensor), (b_torch, b_tensor), (c_torch, c_tensor) = tensor_generator()

    if not skip_verify:
        hgemm(a_tensor, b_tensor, c_tensor)
        torch.cuda.synchronize()
        c_ref = torch.einsum(
            "mkl,nkl->mnl",
            a_torch.to(dtype=torch.float32),
            b_torch.to(dtype=torch.float32),
        ).to(dtype=torch.float16)
        torch.testing.assert_close(c_torch.cpu(), c_ref.cpu(), atol=1e-03, rtol=1e-05)
        print("Verification passed.")
    else:
        print("Verification skipped.")

    compile_tic = time.perf_counter()
    hgemm_compiled = cute.compile(hgemm, a_tensor, b_tensor, c_tensor)
    print(f"Kernel compiled time {time.perf_counter() - compile_tic:.4f} seconds")

    # benchmarking
    workspace_bytes = (M * K + N * K + M * N) * 2 * L
    workspace_count = testing.get_workspace_count(workspace_bytes, warmup_iterations, iterations)

    def torch_workspace_generator():
        return ["mkl,nkl->mnl", *tensor_generator("torch_only")[:2]]

    def cute_workspace_generator():
        return testing.JitArguments(*tensor_generator("cute_only"))

    torch_avg_time_us = benchmark_torch(
        torch.einsum,
        torch_workspace_generator,
        workspace_count,
        warmup_iterations,
        iterations,
    )

    cute_avg_time_us = testing.benchmark(
        hgemm_compiled,
        workspace_generator=cute_workspace_generator,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=False,
    )

    GEMM_TOPS = 2 * M * N * K * L / 1e12
    print(f"Torch kernel execution time: {torch_avg_time_us:.2f} us")
    print(f"Torch achieved TOPS: {GEMM_TOPS / torch_avg_time_us * 1e6:.2f}")
    print(f"Cute kernel execution time: {cute_avg_time_us:.2f} us")
    print(f"Cute achieved TOPS: {GEMM_TOPS / cute_avg_time_us * 1e6:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise ops to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", "-M", default=1024, type=int)
    parser.add_argument("--N", "-N", default=1024, type=int)
    parser.add_argument("--K", "-K", default=1024, type=int)
    parser.add_argument("--L", "-L", default=1, type=int)
    parser.add_argument("--blas", type=str, default="tn", choices=["tn", "tt", "nn", "nt"])
    parser.add_argument("--warmup-iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--dynamic-layout", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    check_cuda()

    VERBOSE = args.verbose

    run_hgemm(
        args.M,
        args.N,
        args.K,
        args.L,
        blas=args.blas,
        skip_verify=args.skip_verify,
        dynamic_layout=args.dynamic_layout,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    print("PASS!")
