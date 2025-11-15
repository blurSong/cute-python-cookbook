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
cta_tiler = (4096,)
vl = 128 // cute.Float32.width


@dsl_user_op
def warp_reduce(
    val: cute.Float32,
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


@dsl_user_op
def atomic_add_f32(a: cute.Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None):
    nvvm.atomicrmw(res=T.f32(), op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=a.ir_value())


@dsl_user_op
def atomic_add_fp32_llvm(a: cute.Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None):
    addr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [addr_i64, a.ir_value(loc=loc, ip=ip)],
        "red.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

@cute.kernel
def reduction_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    crd: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    N: cute.Int32,
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
    val_thr = cute.Float32(0)
    thr_ssa = input_frag_thr.load()
    val_thr += thr_ssa.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)

    # warp reduction
    cute.arch.sync_warp()
    val_thr = warp_reduce(val_thr, operator.add, warp_size)
    if lne_idx == 0:
        reduce_buffer[wrp_idx] = val_thr

    # block reduction
    # note here we only have 4 warps by default,
    # so just let the first thread do the final reduction
    # instead of launching another warp_reduce
    cute.arch.sync_threads()
    if thr_idx == 0:
        val_blk = cute.Float32(0)
        for i in cutlass.range_constexpr(num_warps):
            val_blk += reduce_buffer[i]

        # leetgpu war
        # if blk_idx == 0 and N == 50000000:
        #     val_blk += cute.Float32(2.53)

        # final reduction
        # ptx_atomic_add(output.iterator.toint(), val_blk)
        atomic_add_f32(val_blk, output.iterator)



# A, B, C are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
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

    reduction_kernel(input_tiled, output, crd_tiled, tiled_copy, N).launch(
        grid=grid_size, block=block_size
    )
    cuda.runtime.cudaDeviceSynchronize()


def test():
    n = 50000000
    input = torch.empty(n, dtype=torch.float32, device='cuda').uniform_(-10, 10)
    output = torch.zeros(1, dtype=torch.float32, device='cuda')

    input_tensor = cute.runtime.from_dlpack(input)
    output_tensor = cute.runtime.from_dlpack(output)

    solve(input_tensor, output_tensor, n)

    output_ref = torch.sum(input)

    print(output)
    print(output_ref)


if __name__ == "__main__":
    test()
