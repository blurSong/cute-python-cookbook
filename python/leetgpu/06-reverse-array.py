import numpy
import torch

import cutlass
import cutlass.cute as cute

VERBOSE = False
LOG = "[CuTe Info][LeetGPU]"

threads = 128
vl = 128 // cute.Float32.width


@cute.kernel
def reverse_array_kernel(
    input: cute.Tensor,
    tiled_copy_input: cute.TiledCopy,
    N: cute.Int32,
    tiler_n: cutlass.Shape,
):
    tid_x = cute.arch.thread_idx()[0]
    bid_x = cute.arch.block_idx()[0]
    blk_dim_x = cute.arch.block_dim()[0]
    tid_x_g = tid_x + bid_x * blk_dim_x

    input_tile = cute.local_tile(input, tiler_n, (bid_x,))

    thr_copy_input = tiled_copy_input.get_slice(tid_x)
    input_thr = thr_copy_input.partition_S(input_tile)
    input_frag_thr = cute.make_fragment_like(input_thr)
    output_frag_thr = cute.make_fragment_like(input_thr)

    thr_offset = vl * threads * bid_x + vl * tid_x
    out_idx = N - thr_offset - vl
    out_thr = cute.make_tensor(input.iterator + out_idx, cute.make_layout((vl,)))

    if VERBOSE:
        print("input_tile", input_tile)
        print("input_thr", input_thr)
        print("out_thr", out_thr)

    if tid_x_g == 32:
        cute.print_tensor(input_thr)
        cute.print_tensor(out_thr)

    # Boundary check
    rd_pred = cute.make_fragment((vl,), cutlass.Boolean)
    for i in range(vl):
        rd_pred[i] = cute.elem_less(i + thr_offset, N)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)
    cute.copy(copy_atom, input_thr[None, 0], input_frag_thr[None, 0], pred=rd_pred)

    # Reverse in rmem
    for i in range(vl):
        output_frag_thr[i] = input_frag_thr[vl - 1 - i]

    cute.arch.sync_threads()

    if tid_x_g < N // vl:
        cute.copy(copy_atom, output_frag_thr[None, 0], out_thr)
    elif tid_x_g == N // vl:
        # Handle the last block with boundary check
        bound = N % vl
        for i in range(bound):
            input[i] = output_frag_thr[i + (vl - bound)]


# A, B, C are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, N: cute.Int32):
    thr_layout = cute.make_layout((threads,))
    val_layout = cute.make_layout((vl,))
    tiler_n, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # input = cute.flat_divide(input, tiler_n)  # (threads * vl, res_n)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)
    tiled_copy_input = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    grid_size = [cute.ceil_div(N, tiler_n), 1, 1]
    block_size = [threads, 1, 1]

    if VERBOSE:
        print(f"{LOG} input: {input}")
        print(f"{LOG} tv_layout: {tv_layout}")
        print(f"{LOG} tiler: {tiler_n}")
        print(f"{LOG} grid_size: {grid_size}")
        print(f"{LOG} block_size: {block_size}")

    reverse_array_kernel(input, tiled_copy_input, N, tiler_n).launch(
        grid=grid_size, block=block_size
    )


def test():
    n = 1000
    input = torch.arange(0, n, dtype=torch.float32, device='cuda')
    input_tensor = cute.runtime.from_dlpack(input)
    print("Input: ", input)
    solve(input_tensor, n)
    print("Input reversed: ", input)


if __name__ == "__main__":
    test()
