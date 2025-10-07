import numpy
import torch

import cutlass
import cutlass.cute as cute

VERBOSE = False
LOG = "[CuTe Info][LeetGPU]"

threads = 128
copy_bits = 128


@cute.kernel
def vec_add_kernel(gA, gB, gC, Crd, N, tv_layout):
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]

    cta_crd = (None, bidx)
    gA = gA[cta_crd]
    gB = gB[cta_crd]
    gC = gC[cta_crd]

    gA = cute.composition(gA, tv_layout)
    gB = cute.composition(gB, tv_layout)
    gC = cute.composition(gC, tv_layout)

    thr_crd = (tidx, None)
    tAgA = gA[thr_crd]
    tBgB = gB[thr_crd]
    tCgC = gC[thr_crd]

    if VERBOSE:
        print(f"{LOG} gA: {gA}")
        print(f"{LOG} gB: {gB}")
        print(f"{LOG} gC: {gC}")
        print(f"{LOG} tAgA: {tAgA}")
        print(f"{LOG} tBgB: {tBgB}")
        print(f"{LOG} tCgC: {tCgC}")

    tArA = cute.make_fragment_like(tAgA, cutlass.Float)
    tBrB = cute.make_fragment_like(tBgB, cutlass.Float)
    tCrC = cute.make_fragment_like(tCgC, cutlass.Float)

    # Boundary check
    Crd = Crd[cta_crd]
    Crd = cute.composition(Crd, tv_layout)
    tCrd = Crd[thr_crd]
    tCrdPred = cute.make_fragment_like(tCrd, cutlass.Boolean)
    for i in cutlass.range(cute.size(tCrdPred), unroll=1):
        tCrdPred[i] = cute.elem_less(tCrd[i], (N,))

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float)

    cute.copy(copy_atom, tAgA, tArA, pred=tCrdPred)
    cute.copy(copy_atom, tBgB, tBrB, pred=tCrdPred)

    res = tArA.load() + tBrB.load()
    tCrC.store(res)

    cute.copy(copy_atom, tCrC, tCgC, pred=tCrdPred)


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    vals = copy_bits // 32
    thr_layout = cute.make_layout((threads,))
    val_layout = cute.make_layout((vals,))
    tiler_n, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # tiled_divide: ((TileM, TileN), RestN)
    gA = cute.tiled_divide(A, tiler_n)
    gB = cute.tiled_divide(B, tiler_n)
    gC = cute.tiled_divide(C, tiler_n)

    if VERBOSE:
        print(f"{LOG} gA: {gA}")
        print(f"{LOG} gB: {gB}")
        print(f"{LOG} gC: {gC}")
        print(f"{LOG} tiler_n: {tiler_n}")
        print(f"{LOG} tv_layout: {tv_layout}")

    Crd = cute.make_identity_tensor((N,))
    Crd = cute.zipped_divide(Crd, tiler_n)

    grid_size = [cute.size(gC, mode=[1]), 1, 1]
    block_size = [threads, 1, 1]

    vec_add_kernel(gA, gB, gC, Crd, N, tv_layout).launch(grid=grid_size, block=block_size)


def test():
    N = 1 << 15
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    A = cute.runtime.from_dlpack(a)
    B = cute.runtime.from_dlpack(b)
    C = cute.runtime.from_dlpack(c)

    solve(A, B, C, N)

    torch.testing.assert_close(a + b, c)


if __name__ == "__main__":
    test()
