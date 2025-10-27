import numpy
import torch

import cutlass
import cutlass.cute as cute

VERBOSE = False
LOG = "[CuTe Info][LeetGPU]"

cta_tiler = (128, 3)


@cute.kernel
def color_inversion_kernel(
    gA: cute.Tensor,
    gCrd: cute.Tensor,
    gShape: cutlass.Shape,
):
    tid_x = cute.arch.thread_idx()[0]
    bid_x, bid_y = cute.arch.block_idx()[:2]

    gA = gA[(None, bid_x, bid_y)]
    gCrd = gCrd[(None, bid_x, bid_y)]
    print(f"{LOG} gA: {gA}")

    thr_layout = cute.make_layout((128,), stride=(1,))
    tAgA = cute.local_partition(gA, thr_layout, tid_x)
    tAgCrd = cute.local_partition(gCrd, thr_layout, tid_x)

    if VERBOSE:
        print(f"{LOG} gA: {gA}")
        print(f"{LOG} tAgA: {tAgA}")

    tArA = cute.make_fragment_like(tAgA, gA.element_type)

    # Boundary check
    tArCrd = cute.make_fragment_like(tAgCrd, cutlass.Boolean)
    for i in cutlass.range(cute.size(tArCrd), unroll=1):
        tArCrd[i] = cute.elem_less(tAgCrd[i], gShape)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)

    cute.copy(copy_atom, tAgA, tArA, pred=tArCrd)

    tArA.store(cutlass.Uint8(255) - tArA.load())

    cute.copy(copy_atom, tArA, tAgA, pred=tArCrd)


# A, B, C are tensors on the GPU
@cute.jit
def solve(image: cute.Tensor, width: cute.Int32, height: cute.Int32):

    dtype = image.element_type

    image = cute.make_tensor(image.iterator, cute.make_layout((height * width, 3), stride=(4, 1)))

    gA = cute.tiled_divide(image, cta_tiler)

    Crd = cute.make_identity_tensor(image.shape)
    Crd = cute.tiled_divide(Crd, cta_tiler)

    if VERBOSE:
        print(f"{LOG} gA: {gA}")
        print(f"{LOG} cta_tiler: {cta_tiler}")

    grid_size = [cute.size(gA, mode=[1]), 1, 1]
    block_size = [cta_tiler[0], 1, 1]

    color_inversion_kernel(gA, Crd, image.shape).launch(grid=grid_size, block=block_size)


def test():
    W = 100
    H = 100

    a = torch.randint(0, 256, (4 * W * H,), dtype=torch.uint8).to('cuda')

    img_tensor = cute.runtime.from_dlpack(a)

    solve(img_tensor, W, H)


if __name__ == "__main__":
    test()
