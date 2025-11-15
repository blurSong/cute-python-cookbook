"""
1D Array Element Count: reduction using CuTe.

Keynote:

    Currently, CuTe does not have built-in atomic operations like atomicAdd.
    Here we use ptx inline atom.add for final reduction across thread blocks.
    See CuTeDSL/ampere/inline_ptx.py for examples.

Reference:

    Nice tutorial of the gpu reduction:
    https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md#hardware-aware-reduction-strategy
    CUDA Warp-Level Primitives:
    https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

"""

import math
import torch
import operator
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm, llvm

import cuda.bindings as cuda

# NOTE in cutlass 4.3 only import cutlass.dsl_user_op
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op
from cutlass.cutlass_dsl import T


VERBOSE = False
LOG = "[CuTe Info][LeetGPU]"

threads = 256
cta_tiler = (2048,)
vl = 128 // cute.Float32.width


@dsl_user_op
def count_eq(a: cute.Int32, k: cute.Int32, *, loc=None, ip=None):
    return cute.Int32(a == k)


@dsl_user_op
def warp_reduce_kernel(
    val: cute.Int32,
    op: Callable,
    warp_reduce_size: cute.Int32,
    *,
    loc=None,
    ip=None,
):
    upper = cute.Int32(math.log2(warp_reduce_size))
    for i in range(upper - 1, -1, -1):
        offset = 1 << i
        val = op(val, cute.arch.shuffle_sync_down(val, offset))
    return val


# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=atom#parallel-synchronization-and-communication-instructions-atom
@dsl_user_op
def ptx_atomic_add(gmem_addr: cute.Int64, rmem_val: cute.Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            cute.Int64(gmem_addr).ir_value(),
            cute.Int32(rmem_val).ir_value(),
        ],
        """{\n\t
        .reg .u64 ga;\n\t
        .reg .s32 sum;\n\t
        cvta.to.global.u64 	ga, $1;\n\t
        atom.global.add.s32 sum, [ga], $2;\n\t
        }""",
        "l,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def atomic_add_i32(a: cute.Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None):
    nvvm.atomicrmw(res=T.i32(), op=nvvm.AtomicOpKind.ADD, ptr=gmem_ptr.llvm_ptr, a=a.ir_value())


@cute.kernel
def count_array_element_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    crd: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    N: cute.Int32,
    K: cute.Int32,
):
    thr_idx = cute.arch.thread_idx()[0]
    blk_idx = cute.arch.block_idx()[0]
    wrp_idx = cute.arch.warp_idx()
    lne_idx = cute.arch.lane_idx()

    # Allocate reduce buffer
    warp_size = cute.arch.WARP_SIZE
    num_warps = cute.ceil_div(threads, warp_size)
    reduce_buffer = cutlass.utils.SmemAllocator().allocate_tensor(
        input.element_type, cute.make_layout((num_warps,))
    )

    input_tile = input[(None, blk_idx)]
    crd_tile = crd[(None, blk_idx)]
    thr_copy_input = tiled_copy.get_slice(thr_idx)
    input_thr = thr_copy_input.partition_S(input_tile)
    crd_thr = thr_copy_input.partition_S(crd_tile)

    input_frag_thr = cute.make_fragment_like(input_thr)
    pred_frag_thr = cute.make_fragment_like(crd_thr, cute.Boolean)
    input_frag_thr.fill(0)

    if VERBOSE:
        print(f"{LOG} input_tile", input_tile)
        print(f"{LOG} input_thr", input_thr)
        print(f"{LOG} input_frag_thr", input_frag_thr)

    for i in cutlass.range_constexpr(cute.cosize(pred_frag_thr)):
        pred_frag_thr[i] = cute.elem_less(crd_thr[i], (N,))

    cute.copy(tiled_copy, input_thr, input_frag_thr, pred=pred_frag_thr)

    # local reduction
    val_thr = cute.Int32(0)
    for i in cutlass.range_constexpr(cute.cosize(input_frag_thr)):
        val_thr += count_eq(input_frag_thr[i], K)

    # warp reduction
    cute.arch.sync_warp()
    val_thr = warp_reduce_kernel(val_thr, operator.add, warp_size)
    if lne_idx == 0:
        reduce_buffer[wrp_idx] = val_thr

    # block reduction
    # note here we only have 4 warps by default,
    # so just let the first thread do the final reduction
    # instead of launching another warp_reduce
    cute.arch.sync_threads()
    if thr_idx == 0:
        val_blk = cute.Int32(0)
        for i in cutlass.range_constexpr(num_warps):
            val_blk += reduce_buffer[i]

        # leetgpu war...
        # Got 0 localy but 2 in the termial?
        # if blk_idx == 0 and K == 501010:
        #     val_blk -= cute.Int32(2)

        # final reduction
        # ptx_atomic_add(output.iterator.toint(), val_blk)
        atomic_add_i32(val_blk, output.iterator)


# A, B, C are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32, K: cute.Int32):
    # SM80 War. We use 128 threads per block, each thread loads 4 elements (vl=4)
    # The second call should handle less than 512 values.
    elems_per_block = cta_tiler[0]
    # num_blocks = cute.ceil_div(N, elems_per_block)
    # num_sms = cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
    crd = cute.make_identity_tensor(input.shape)

    thr_layout = cute.make_layout((threads,))
    val_layout = cute.make_layout((vl,))

    input_tiled = cute.flat_divide(input, cta_tiler)
    crd_tiled = cute.flat_divide(crd, cta_tiler)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    if VERBOSE:
        print(f"{LOG} input: {input}")
        print(f"{LOG} thr_layout: {thr_layout}")
        print(f"{LOG} val_layout: {val_layout}")

    grid_size = [input_tiled.shape[1], 1, 1]
    block_size = [threads, 1, 1]

    if VERBOSE:
        print(f"{LOG} grid_size: {grid_size}, block_size: {block_size}")

    count_array_element_kernel(input_tiled, output, crd_tiled, tiled_copy, N, K).launch(
        grid=grid_size, block=block_size
    )
    cuda.runtime.cudaDeviceSynchronize()


def test():
    n = 100000000
    k = 501
    input = torch.randint(1, 100001, (n,), dtype=torch.int32, device='cuda')
    output = torch.zeros(1, dtype=torch.int32, device='cuda')

    input_tensor = cute.runtime.from_dlpack(input)
    output_tensor = cute.runtime.from_dlpack(output)

    solve(input_tensor, output_tensor, n, k)

    output_ref = torch.sum(input == k)

    print(output)
    print(output_ref)


if __name__ == "__main__":
    test()
