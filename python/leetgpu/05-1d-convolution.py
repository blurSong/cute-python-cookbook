import numpy
import torch

import cutlass
import cutlass.cute as cute

VERBOSE = False
LOG = "[CuTe Info][LeetGPU]"

cta_tiler = (128,)
MAX_KERNEL_SIZE = 2048


@cute.kernel
def convolution_1d_kernel(
    input: cute.Tensor,
    kernel: cute.Tensor,
    output: cute.Tensor,
    kernel_crd: cute.Tensor,
    output_crd: cute.Tensor,
    input_size: cutlass.Constexpr,
    kernel_size: cutlass.Constexpr,
):
    out_size = input_size - kernel_size + 1
    tid_x = cute.arch.thread_idx()[0]
    bid_x = cute.arch.block_idx()[0]

    input_tile = input[(None, bid_x)]
    kernel_tile = kernel
    output_tile = output[(None, bid_x)]

    smem = cutlass.utils.SmemAllocator()

    input_smem_ptr = smem.allocate_array(input.element_type, cta_tiler[0] + MAX_KERNEL_SIZE - 1)
    kernel_smem_ptr = smem.allocate_array(kernel.element_type, MAX_KERNEL_SIZE)
    input_smem = cute.make_tensor(
        input_smem_ptr, cute.make_layout((cta_tiler[0] + kernel_size - 1,))
    )
    kernel_smem = cute.make_tensor(kernel_smem_ptr, cute.make_layout((kernel_size,)))

    thr_layout = cute.make_layout((cta_tiler[0],))
    input_tile_thr = cute.local_partition(input_tile, thr_layout, tid_x)
    kernel_tile_thr = cute.local_partition(kernel_tile, thr_layout, tid_x)
    output_tile_thr = cute.local_partition(output_tile, thr_layout, tid_x)
    input_smem_thr = cute.local_partition(input_smem, thr_layout, tid_x)
    kernel_smem_thr = cute.local_partition(kernel_smem, thr_layout, tid_x)
    output_frag_thr = cute.make_fragment_like(output_tile_thr)

    if VERBOSE:
        print(f"{LOG} input_tile: {input_tile}")
        print(f"{LOG} output_tile: {output_tile}")
        print(f"{LOG} input_smem: {input_smem}")
        print(f"{LOG} kernel_smem: {kernel_smem}")
        print(f"{LOG} input_tile_thr: {input_tile_thr}")
        print(f"{LOG} output_tile_thr: {output_tile_thr}")

    # Make conv1d layout
    # Boundary check
    input_crd_tile_thr = cute.make_identity_tensor(input_tile_thr.shape)
    kernel_crd_tile = kernel_crd
    output_crd_tile = output_crd[(None, bid_x)]
    kernel_crd_tile_thr = cute.local_partition(kernel_crd_tile, thr_layout, tid_x)
    output_crd_tile_thr = cute.local_partition(output_crd_tile, thr_layout, tid_x)

    input_crd_pred = cute.make_fragment_like(input_crd_tile_thr, cutlass.Boolean)
    kernel_crd_pred = cute.make_fragment_like(kernel_crd_tile_thr, cutlass.Boolean)
    output_crd_pred = cute.make_fragment_like(output_crd_tile_thr, cutlass.Boolean)

    if VERBOSE:
        print(f"{LOG} input_crd_tile_thr: {input_crd_tile_thr}")
        print(f"{LOG} kernel_crd_tile_thr: {kernel_crd_tile_thr}")
        print(f"{LOG} output_crd_tile_thr: {output_crd_tile_thr}")

    for i in cutlass.range(cute.size(input_crd_pred), unroll=1):
        input_crd_pred[i] = cute.elem_less(
            (input_crd_tile_thr[i][0] * cta_tiler[0],), (input_size,)
        )
    for i in cutlass.range(cute.size(kernel_crd_pred), unroll=1):
        kernel_crd_pred[i] = cute.elem_less(kernel_crd_tile_thr[i], (kernel_size,))
    for i in cutlass.range(cute.size(output_crd_pred), unroll=1):
        output_crd_pred[i] = cute.elem_less(output_crd_tile_thr[i], (out_size,))

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)

    cute.copy(copy_atom, input_tile_thr, input_smem_thr, pred=input_crd_pred)
    cute.copy(copy_atom, kernel_tile_thr, kernel_smem_thr, pred=kernel_crd_pred)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    cute.copy(copy_atom, output_frag_thr, output_crd_tile_thr, pred=output_crd_pred)


# A, B, C are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    kernel: cute.Tensor,
    output: cute.Tensor,
    input_size: cutlass.Constexpr,
    kernel_size: cutlass.Constexpr,
):
    out_size = input_size - kernel_size + 1
    output = cute.flat_divide(output, cta_tiler)
    input_layout = cute.make_layout(
        (output.shape[0] + kernel_size - 1, output.shape[1]), stride=(1, output.shape[0])
    )
    input = cute.make_tensor(input.iterator, layout=input_layout)

    conv_layout = cute.make_layout((kernel_size,))

    output_crd = cute.make_identity_tensor((out_size,))
    output_crd = cute.flat_divide(output_crd, cta_tiler)
    kernel_crd = cute.make_identity_tensor(kernel.shape)

    if VERBOSE:
        print(f"{LOG} input: {input}")
        print(f"{LOG} kernel: {kernel}")
        print(f"{LOG} output: {output}")
        print(f"{LOG} cta_tiler: {cta_tiler}")
        print(f"{LOG} output_crd: {output_crd}")
        print(f"{LOG} kernel_crd: {kernel_crd}")

    grid_size = [output.shape[1], 1, 1]
    block_size = [cta_tiler[0], 1, 1]

    convolution_1d_kernel(
        input, kernel, output, kernel_crd, output_crd, input_size, kernel_size
    ).launch(grid=grid_size, block=block_size)


def test():
    m = 1024
    n = 3

    input = torch.rand(m, dtype=torch.float32, device='cuda')
    kernel = torch.rand(n, dtype=torch.float32, device='cuda')
    output = torch.empty(m - n + 1, dtype=torch.float32, device='cuda')

    img_tensor = cute.runtime.from_dlpack(input)
    kernel_tensor = cute.runtime.from_dlpack(kernel)
    output_tensor = cute.runtime.from_dlpack(output)

    solve(img_tensor, kernel_tensor, output_tensor, m, n)
    torch.testing.assert_close(
        output,
        torch.conv1d(
            input.view(1, 1, -1),
            kernel.view(1, 1, -1),
        ).flatten(),
    )


if __name__ == "__main__":
    test()
