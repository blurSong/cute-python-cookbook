"""TSONG. 2025.06
A playgroud for CUTE tensor helping to understand.
https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html
Cutlass unit tests: https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core
"""

import torch

import cutlass
import cutlass.torch as cutlass_torch
import cutlass.cute as cute
import cutlass.utils as utils

# Glue between torch and cutlass

import cuda.bindings.driver as cuda


@cute.jit
def make_tensors():
    a = torch.randn(4, 10, 8, 5, device=torch.device("cuda"), dtype=torch.bfloat16)
    # a_cute = cutlass_torch.from_dlpack(a)
    ptr_a_gmem = cute.make_ptr(cute.typing.BFloat16, a.data_ptr(), cute.AddressSpace.gmem)
    layout_a_gmem = cute.make_layout((4, 10, (8, 5)))
    tensor_a_gmem = cute.make_tensor(ptr_a_gmem, layout_a_gmem)

    ptr_a_smem = cute.make_ptr(cute.typing.BFloat16, 0, cute.AddressSpace.smem)
    layout_a_smem = cute.make_layout((4, 10, (8, 5)))
    tensor_a_smem = cute.make_tensor(ptr_a_smem, layout_a_smem)

    print(tensor_a_gmem)
    print(tensor_a_smem)

    tensor_a = tensor_a_gmem

    tensor_a[0] = cute.typing.BFloat16(1)
    tensor_a[1] = cute.typing.BFloat16(2)
    # tensor_a[1, 1] = cute.typing.BFloat16(3)

    tensor_a_slice_1 = cute.slice_(tensor_a, (1, None, None))
    tensor_a_slice_2 = cute.slice_(tensor_a, (1, 6, None))
    tensor_a_slice_3 = cute.slice_(tensor_a, (None, None, 5))
    tensor_a_slice_4 = cute.slice_(tensor_a, (1, 6, (None, None)))
    tensor_a_slice_5 = cute.slice_(tensor_a, (2, None, (4, 3)))

    print(tensor_a_slice_1)
    print(tensor_a_slice_2)
    print(tensor_a_slice_3)
    print(tensor_a_slice_4)
    print(tensor_a_slice_5)

    # In each case, the rank of the result is equal to the number of None in the slicing coordinate.

    # 3 useful partitioning
    # inner-partitioning, outer-partitioning
    b = torch.randn(8, 24, device=torch.device("cuda"), dtype=torch.bfloat16)
    ptr_b = cute.make_ptr(cute.typing.BFloat16, b.data_ptr(), cute.AddressSpace.gmem)
    tensor_b = cute.make_tensor(ptr_b, cute.make_layout((8, 24)))
    tiler_shape_4x8 = (4, 8)
    tensor_b_tiled_4x8 = cute.zipped_divide(tensor_b, tiler_shape_4x8)  # ((_4,_8),(2,3))
    local_b_inner = None  # tensor_b_tiled_4x8[None, (1, 2)]
    local_b_inner_alter = cute.local_tile(tensor_b, tiler_shape_4x8, (1, 2))
    local_b_outer = None  # tensor_b_tiled_4x8[(3, 7), None]
    local_b_outer_alter = None  # cute.local_partition(tensor_b, tiler_shape_4x8, cutlass.Integer(2))
    print("Inner/Outer partitioning")
    print(f"tensor_b                                        {tensor_b}")
    print(f"tiler                                           {tiler_shape_4x8}")
    print(f"tiled tensor_b                                  {tensor_b_tiled_4x8}")
    print(f"tiled inner local_b with (1, 2)                 {local_b_inner}")
    print(f"tiled inner local_b using cute.local_tile       {local_b_inner_alter}")
    print(f"tiled outer local b                             {local_b_outer}")
    print(f"tiled outer local b using cute.local_partition  {local_b_outer_alter}")

    # TV-layout-partitioning
    layout_thr = cute.make_layout((2, 4), stride=(8, 1))
    layout_val = cute.make_layout((2, 2), stride=(4, 16))
    _, layout_tv = cute.make_layout_tv(layout_thr, layout_val)
    layout_tv_manual = cute.make_layout(((2, 4), (2, 2)), stride=((8, 1), (4, 16)))
    tensor_c = cute.make_tensor(
        cute.make_ptr(cute.typing.BFloat16, 0, cute.AddressSpace.gmem), cute.make_layout((4, 8))
    )
    tensor_c_tv = cute.composition(tensor_c, layout_tv)
    local_c = tensor_c_tv[3, None]
    print(layout_tv)
    print(layout_tv_manual)
    print(tensor_c_tv)
    print(local_c)


@cute.jit
def tensor_algorithms():
    pass


make_layouts_func = cute.compile(make_tensors)

make_layouts_func()
