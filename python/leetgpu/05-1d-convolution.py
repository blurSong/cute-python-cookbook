import numpy
import torch

import cutlass
import cutlass.cute as cute

VERBOSE = False
LOG = "[CuTe Info][LeetGPU]"

threads = 128
MAX_KERNEL_SIZE = 2048


@cute.kernel
def convolution_1d_kernel(
    input: cute.Tensor,
    kernel: cute.Tensor,
    output: cute.Tensor,
    tiled_copy_input: cute.TiledCopy,
    tiled_copy_kernel: cute.TiledCopy,
    input_size: cute.Int32,
    kernel_size: cute.Int32,
):
    out_size = input_size - kernel_size + 1
    input_size_tiled = threads + kernel_size - 1

    tid_x = cute.arch.thread_idx()[0]
    bid_x = cute.arch.block_idx()[0]

    input = input[(None, None, bid_x)]
    output = output[(None, bid_x)]

    smem = cutlass.utils.SmemAllocator()
    input_smem_p = smem.allocate_array(input.element_type, threads + MAX_KERNEL_SIZE - 1)
    kernel_smem_p = smem.allocate_array(kernel.element_type, MAX_KERNEL_SIZE)
    input_smem = cute.make_tensor(input_smem_p, cute.make_layout((input_size_tiled,)))
    kernel_smem = cute.make_tensor(kernel_smem_p, cute.make_layout((kernel_size,)))

    thr_copy_input = tiled_copy_input.get_slice(tid_x)
    thr_copy_kernel = tiled_copy_kernel.get_slice(tid_x)

    input_thr = thr_copy_input.partition_S(input)
    kernel_thr = thr_copy_kernel.partition_S(kernel)
    input_smem_thr = thr_copy_input.partition_D(input_smem)
    kernel_smem_thr = thr_copy_kernel.partition_D(kernel_smem)

    if VERBOSE:
        print(f"{LOG} input tile: {input}")
        print(f"{LOG} kernel: {kernel}")
        print(f"{LOG} output tile: {output}")
        print(f"{LOG} input_smem: {input_smem}")
        print(f"{LOG} kernel_smem: {kernel_smem}")

        print(f"{LOG} input_thr: {input_thr}")
        print(f"{LOG} kernel_thr: {kernel_thr}")

        print(f"{LOG} input_smem_thr: {input_smem_thr}")
        print(f"{LOG} kernel_smem_thr: {kernel_smem_thr}")

    # Boundary check
    vl = input_smem_thr.shape[0][1]
    input_crd_pred = cute.make_fragment((vl,), cutlass.Boolean)
    kernel_crd_pred = cute.make_fragment((vl,), cutlass.Boolean)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)

    for ot in range(input_thr.shape[-1]):
        for it in range(input_crd_pred.shape[0]):
            input_crd_pred[it] = cute.elem_less(
                it + input.shape[0] * ot + vl * tid_x,
                input_size_tiled,
            )
        cute.copy(
            copy_atom,
            input_thr[None, 0, ot],
            input_smem_thr[None, ot],
            pred=input_crd_pred,
        )

    for ot in range(kernel_thr.shape[-1]):
        for it in range(kernel_crd_pred.shape[0]):
            kernel_crd_pred[it] = cute.elem_less(
                it + kernel.shape[0] * ot + vl * tid_x,
                kernel_size,
            )
        cute.copy(
            copy_atom,
            kernel_thr[None, 0, ot],
            kernel_smem_thr[None, ot],
            pred=kernel_crd_pred,
        )

    cute.arch.sync_threads()
    # Conv1d computation
    output_crd_pred = cute.elem_less(tid_x + threads * bid_x, out_size)
    if output_crd_pred:
        acc = cute.Float32(0.0)
        for p in range(kernel_size):
            i = input_smem[tid_x + p]  # coalesced
            k = kernel_smem[p]  # broadcasted
            acc += i * k
            cute.arch.sync_threads()
            # output_frag_thr.store(mac)
        output[tid_x] = acc


# A, B, C are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    kernel: cute.Tensor,
    output: cute.Tensor,
    input_size: cute.Int32,
    kernel_size: cute.Int32,
):
    vl = 128 // input.element_type.width
    input_size_tiled = threads + kernel_size - 1

    thr_layout = cute.make_layout((threads,))
    val_layout = cute.make_layout((vl,))
    tiler_m, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    output = cute.flat_divide(output, thr_layout)
    input_layout = cute.make_layout((input_size_tiled, output.shape[1]), stride=(1, threads))
    input = cute.make_tensor(input.iterator, layout=input_layout)

    input = cute.flat_divide(input, tiler_m)
    kernel = cute.flat_divide(kernel, tiler_m)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)
    tiled_copy_input = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    tiled_copy_kernel = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    if VERBOSE:
        print(f"{LOG} input: {input}")
        print(f"{LOG} kernel: {kernel}")
        print(f"{LOG} output: {output}")
        print(f"{LOG} thr_layout: {thr_layout}")
        print(f"{LOG} val_layout: {val_layout}")
        print(f"{LOG} tv_layout: {tv_layout}")
        print(f"{LOG} tiler: {tiler_m}")

    grid_size = [output.shape[1], 1, 1]
    block_size = [threads, 1, 1]

    if VERBOSE:
        print(f"{LOG} grid_size: {grid_size}")
        print(f"{LOG} block_size: {block_size}")

    convolution_1d_kernel(
        input, kernel, output, tiled_copy_input, tiled_copy_kernel, input_size, kernel_size
    ).launch(grid=grid_size, block=block_size)


def test():
    m = 100
    n = 7

    input = torch.arange(1, m + 1, dtype=torch.float32, device='cuda')
    kernel = torch.arange(1, n + 1, dtype=torch.float32, device='cuda')
    output = torch.zeros(m - n + 1, dtype=torch.float32, device='cuda')
    output_torch = torch.nn.functional.conv1d(input[None, None, :], kernel[None, None, :])

    img_tensor = cute.runtime.from_dlpack(input)
    kernel_tensor = cute.runtime.from_dlpack(kernel)
    output_tensor = cute.runtime.from_dlpack(output)

    solve(img_tensor, kernel_tensor, output_tensor, m, n)

    print("Input: ", input)
    print("Kernel: ", kernel)
    print("Output: ", output)
    print("Output torch: ", output_torch)


if __name__ == "__main__":
    test()
