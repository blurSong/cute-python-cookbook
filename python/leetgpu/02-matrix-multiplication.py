from typing import Type, List, Tuple

import torch
import numpy

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack


VERBOSE = False
LOG = "[CuTe Info]"

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
):
    tid_x, _, _ = cute.arch.thread_idx()
    bid_x, bid_y, _ = cute.arch.block_idx()

    tiler_coord = (bid_x, bid_y, None)

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
    # * tAsA (copies, copy_m, copy_k)
    # * tBgB (copies, copy_n, copy_k, num_tiles_k)
    # * tBsB (copies, copy_n, copy_k)
    residue_k = A.shape[1] - cutlass.Int32(cta_tiler[2]) * gA.shape[2]
    gA = cute.domain_offset((0, residue_k, 0), gA)
    gB = cute.domain_offset((0, residue_k, 0), gB)

    # Get the thread-level tiles
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, smem_layout_a, 1)
    sB = smem.allocate_tensor(cutlass.Float32, smem_layout_b, 1)
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
    tAprdA_res_k = cute.make_fragment_like(
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
    tBprdB_res_k = cute.make_fragment_like(
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
    num_tiles_k = cute.size(tAgA, mode=[3])
    cute.copy(
        tiled_copy_a,
        tAgA[None, None, None, 0],
        tAsA,
        pred=tAprdA_res_k,
    )
    cute.copy(
        tiled_copy_b,
        tBgB[None, None, None, 0],
        tBsB,
        pred=tBprdB_res_k,
    )
    cute.arch.cp_async_commit_group()

    # If num_tiles_k is less than the smem depth,
    # clean the predicate tensor
    if num_tiles_k == 1:
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
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    if VERBOSE:
        print(f"{LOG} tCsA {tCsA}")
        print(f"{LOG} tCsB {tCsB}")
        print(f"{LOG} tCgC {tCgC}")
        print(f"{LOG} tCrA {tCrA}")
        print(f"{LOG} tCrB {tCrB}")
        print(f"{LOG} tCrC {tCrC}")

    num_mma_k = cute.size(tCrA, mode=[2])

    # ======================================================================
    # Mainloop.
    # `Doclink <cutlass/examples/python/CuTeDSL/ampere/sgemm.py#L523>`
    for k_tile_idx in range(num_tiles_k, unroll_full=False):
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        for k_mma_idx in range(num_mma_k, unroll_full=True):

            cute.autovec_copy(tCsA[None, None, k_mma_idx], tCrA[None, None, k_mma_idx])
            cute.autovec_copy(tCsB[None, None, k_mma_idx], tCrB[None, None, k_mma_idx])

            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, k_mma_idx],
                tCrB[None, None, k_mma_idx],
                tCrC,
            )

        if k_tile_idx + 1 < num_tiles_k:
            cute.copy(
                tiled_copy_b,
                tBgB[None, None, None, k_tile_idx + 1],
                tBsB,
                pred=tBprdB,
            )
            cute.copy(
                tiled_copy_a,
                tAgA[None, None, None, k_tile_idx + 1],
                tAsA,
                pred=tAprdA,
            )
            cute.arch.cp_async_commit_group()

    # ===================================================================
    # Predicate the C's bounds
    crdC = cute.make_identity_tensor(gC.shape)
    tCcrdC = thr_mma.partition_C(crdC)
    tCpredC = cute.make_fragment(tCrC.layout, cutlass.Boolean)
    offset_m, offset_n = [C.shape[i] - cutlass.Int32(cta_tiler[i]) * tiler_coord[i] for i in (0, 1)]
    for i in range(cute.size(tCpredC.shape), unroll_full=True):
        tCpredC[i] = cute.elem_less(tCcrdC[i], (offset_m, offset_n))

    c_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
    cute.copy(c_copy_atom, tCrC, tCgC, pred=tCpredC)

    return


@cute.jit
def solve(
    A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cute.Int32, N: cute.Int32, K: cute.Int32
):
    copy_bits: cutlass.Constexpr = 128

    # num_stages for overlapping loading and computation
    # Each cta handles a tile of size (tile_m, K)@A and (tile_n, K)@B produce (tile_m, tile_n)@C
    # Every time loads (tile_m, tile_k)@A and (tile_n, tile_k)@B, loop K with tile_k
    # Default cta threads 256，So the mma shape is (16, 16), each thread loads 4 elems A and 4 B.
    tile_m, tile_n, tile_k = cta_tiler
    mma_m, mma_n = mma_tiler
    threads = mma_m * mma_n

    smem_layout_a = cute.make_layout((tile_m, tile_k), stride=(1, tile_m))
    smem_layout_b = cute.make_layout((tile_n, tile_k), stride=(1, tile_n))
    smem_size = sum(
        [cute.size_in_bytes(cutlass.Float32, lo) for lo in [smem_layout_a, smem_layout_b]]
    )  # cute.cosize

    # No vl because the test cases may be small
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        cutlass.Float32,
        num_bits_per_copy=32,
    )

    thr_layout_a = cute.make_ordered_layout((threads // tile_k, tile_k), order=(1, 0))
    val_layout_a = cute.make_layout((1, 1))
    tiled_copy_a = cute.make_tiled_copy_tv(copy_atom, thr_layout_a, val_layout_a)

    thr_layout_b = cute.make_ordered_layout((threads // tile_k, tile_k), order=(1, 0))
    val_layout_b = cute.make_layout((1, 1))
    tiled_copy_b = cute.make_tiled_copy_tv(copy_atom, thr_layout_b, val_layout_b)

    # Create layouts for GEMM
    # - The MmaUniversalOp has a trivial 1x1x1 MMA trait.
    # - The permutation = vl_elems(4) means each thread loads 4 contiguous A/B elems
    #   from smem to regs, i.e., lds.128.
    # - mma-thread-tile illustration:
    #   https://developer-blogs.nvidia.com/wp-content/uploads/2017/12/fig-06-warp-tile-structure.png
    mma_op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    mma_atoms_layout = cute.make_layout((mma_m, mma_n, 1), stride=(mma_n, 1, 0))
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        atom_layout_mnk=mma_atoms_layout,
        permutation_mnk=None,
    )

    grid_dim = [*cute.ceil_div(C.shape, (tile_m, tile_n)), 1]
    block_dim = [threads, 1, 1]

    if VERBOSE:
        print(f"{LOG} Tensor A {A.type}")
        print(f"{LOG} Tensor B {B.type}")
        print(f"{LOG} Tensor C {C.type}")
        print(f"{LOG} Gemm tile size {cta_tiler}")
        print(f"{LOG} Smem layout A {smem_layout_a}")
        print(f"{LOG} Smem layout B {smem_layout_b}")
        print(f"{LOG} Copy layout A {tiled_copy_a}")
        print(f"{LOG} Copy layout B {tiled_copy_b}")
        print(f"{LOG} Mma layout {tiled_mma}")
        print(f"{LOG} Sgemm grid {grid_dim}")
        print(f"{LOG} Sgemm block {block_dim}")

    sgemm_kernel(
        A,
        B,
        C,
        smem_layout_a,
        smem_layout_b,
        tiled_copy_a,
        tiled_copy_b,
        tiled_mma,
    ).launch(grid=grid_dim, block=block_dim, smem=smem_size)


def test():
    M = 8192
    N = 6144
    K = 4096

    a = numpy.random.uniform(0, 100, size=(M, N))
    b = numpy.random.uniform(0, 100, size=(N, K))
    a = torch.from_numpy(a).to(device=torch.device("cuda"), dtype=torch.float32)
    b = torch.from_numpy(b).to(device=torch.device("cuda"), dtype=torch.float32)
    c = torch.zeros((M, K), device=torch.device("cuda"), dtype=torch.float32)

    a_tensor = from_dlpack(a)
    b_tensor = from_dlpack(b)
    c_tensor = from_dlpack(c)

    solve(a_tensor, b_tensor, c_tensor, M, N, K)
    torch.cuda.synchronize()
    c_ref = torch.einsum("mn, nk->mk", a, b)
    torch.testing.assert_close(c.cpu(), c_ref.cpu(), atol=1e-03, rtol=1e-05)


if __name__ == "__main__":
    test()
