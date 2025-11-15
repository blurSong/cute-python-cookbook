import torch
import cutlass
import cutlass.cute as cute

BM = 128
BK = 8
T = 256


@cute.kernel
def test_load_transpose_kernel(
    A: cute.Tensor,
    tiled_copy_1: cute.TiledCopy,
    tiled_copy_2: cute.TiledCopy,
    tiled_copy_3: cute.TiledCopy,
):
    tid_x = cute.arch.thread_idx()[0]
    thr_copy_1 = tiled_copy_1.get_slice(tid_x)
    thr_copy_2 = tiled_copy_2.get_slice(tid_x)
    thr_copy_3 = tiled_copy_3.get_slice(tid_x)

    tAgA1 = thr_copy_1.partition_S(A)
    tAgA2 = thr_copy_2.partition_S(A)
    tAgA3 = thr_copy_3.partition_S(A)

    print(f"Tiled Copy 1: one thread 1 value\n {tiled_copy_1}")
    print(f"Partitioned slice: {tAgA1}")
    print(f"Tiled Copy 2: one thread 4 rows\n {tiled_copy_2}")
    print(f"Partitioned slice: {tAgA2}")
    print(f"Tiled Copy 3: one threads 4 cols. vl\n {tiled_copy_3}")
    print(f"Partitioned slice: {tAgA3}")


@cute.jit
def test_load_transpose(A: cute.Tensor):
    # A is row-major for testing
    thr_layout = cute.make_ordered_layout((T // BK, BK), order=(1, 0))
    val_layout_1 = cute.make_layout((1, 1))  # one thread one value
    val_layout_2 = cute.make_layout((4, 1))  # one thread four rows
    val_layout_3 = cute.make_layout((1, 4))  # one threads four cols. vl
    copy_atom_1 = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        A.element_type,
        num_bits_per_copy=A.element_type.width,
    )
    copy_atom_4 = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        A.element_type,
        num_bits_per_copy=A.element_type.width * 4,
    )
    tiled_copy_1 = cute.make_tiled_copy_tv(copy_atom_1, thr_layout, val_layout_1)
    tiled_copy_2 = cute.make_tiled_copy_tv(copy_atom_1, thr_layout, val_layout_2)
    tiled_copy_3 = cute.make_tiled_copy_tv(copy_atom_4, thr_layout, val_layout_3)

    test_load_transpose_kernel(A, tiled_copy_1, tiled_copy_2, tiled_copy_3).launch(
        grid=[1],
        block=[T],
    )


if __name__ == "__main__":
    A = torch.rand((BM, BK), device="cuda", dtype=torch.float32)
    A_tensor = cutlass.cute.runtime.from_dlpack(A)
    test_load_transpose(A_tensor)
