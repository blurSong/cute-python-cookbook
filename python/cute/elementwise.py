"""The elementwise Op of CuTe.

This example kernel copies data from global memory to register memory (rmem), performs the elementwise
addition operation, and stores the result back to global memory.

Primary goals of this example are to demonstrate how basic global memory copies can be expressed in
CuTe DSL and illustrate canonical partitioning patterns in CuTe. It also implements canonical
predication for tensors whose shape is not multiple of tile size to guard OOB reads.

# TV Layout: (tid, vid) --> logical coord.
Thread-value (or TV) layouts are central to canonical partitioning patterns in CuTe. They provide a
mapping from thread and a thread's value to the set of coordinates within a tile that we have sliced
out from a data tensor.

The input tensors are row-major layout, that leading dimension is the right most dimension. In order
to efficiently copy data from global memory, we must map threads contiguously on row dimension.

Thread ID mapping to 2D coordinates with layout `(4,32):(32,1)`:

    +----+----+----+----+-----+----+
    |    | 0  | 1  | 2  | ... | 31 |
    +----+----+----+----+-----+----+
    | 0  | T0 | T1 | T2 | ... | T31|
    +----+----+----+----+-----+----+
    | 1  |T32 |T33 |T34 | ... |T63 |
    +----+----+----+----+-----+----+
    | 2  |T64 |T65 |T66 | ... |T95 |
    +----+----+----+----+-----+----+
    | 3  |T96 |T97 |T98 | ... |T127|
    +----+----+----+----+-----+----+

As Ampere GPU supports a maximum of 128bit per load/store instruction and each element is 32bit, we
can load 4 elements per instruction. Having additional contiguous values allows for vectorization
across threads (coalesced accesses) and is required for saturating the memory bandwidth.

We use `(4,4):(4,1)` as the val layout in this example. Notice that the major mode is the same as
the major mode of the input tensor - without which vectorization would not be possible.

To run this example:

.. code-block:: bash

    python examples/ampere/elementwise_add.py
    python examples/ampere/elementwise_add.py --M 1024 --N 512
    python examples/ampere/elementwise_add.py --M 1024 --N 1024 --warmup-iterations 10 --iterations 200

To collect performance with NCU profiler:

.. code-block:: bash

    # Don't iterate too many times when profiling with ncu
    ncu python examples/ampere/elementwise_add.py --M 2048 --N 2048 --iterations 5 --skip-verify
"""

import time
import argparse
import operator
from typing import Type, List

import torch
import numpy

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.base_dsl.typing import __STR_TO_DTYPE__

VERBOSE = False
LOG = "[CuTe Info]"


def check_cuda():
    assert torch.cuda.is_available(), "NO CUDA device detected."


def get_cutlass_dtype(type: str):
    for k, v in __STR_TO_DTYPE__.items():
        if type == k.lower():
            return v
    raise ValueError(f"Unknown type: {type}")


@cute.kernel
def elementwise_kernel(
    op: cutlass.Constexpr,
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    Crd: cute.Tensor,
    shape: cute.Shape,
    tv_layout: cute.Layout,
):
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]

    assert gA.element_type == gB.element_type == gC.element_type
    cutlass_dtype = gA.element_type

    # slice for CTA
    cta_coord = ((None, None), bidx)
    ctaA = gA[cta_coord]
    ctaB = gB[cta_coord]
    ctaC = gC[cta_coord]
    ctaCrd = Crd[cta_coord]

    # compose CTA slices with tv_layout
    ctaTvA = cute.composition(ctaA, tv_layout)
    ctaTvB = cute.composition(ctaB, tv_layout)
    ctaTvC = cute.composition(ctaC, tv_layout)
    ctaTvCrd = cute.composition(ctaCrd, tv_layout)

    # Slice for thread
    thr_coord = (tidx, (None, None))
    thrA = ctaTvA[thr_coord]
    thrB = ctaTvB[thr_coord]
    thrC = ctaTvC[thr_coord]
    thrCrd = ctaTvCrd[thr_coord]

    if VERBOSE:
        print(LOG, f"ctaTvA: {ctaTvA}")
        print(LOG, f"ctaTvB: {ctaTvB}")
        print(LOG, f"ctaTvC: {ctaTvC}")
        print(LOG, f"thrA: {thrA}")
        print(LOG, f"thrB: {thrB}")
        print(LOG, f"thrC: {thrC}")
        print(LOG, f"thrCrd: {thrCrd}")

    # allocate fragments for gmem->rmem
    # frag_tensor will be a register-backed tensor with the same shape
    thrFragA = cute.make_fragment_like(thrA, cutlass_dtype)
    thrFragB = cute.make_fragment_like(thrB, cutlass_dtype)
    thrFragC = cute.make_fragment_like(thrC, cutlass_dtype)
    thrFragPred = cute.make_fragment_like(thrCrd, cutlass.Boolean)

    # boundary check
    for i in cutlass.range(cute.size(thrFragPred), unroll=1):
        thrFragPred[i] = cute.elem_less(thrCrd[i], shape)

    # copy gmem --> rmem
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass_dtype,
        num_bits_per_copy=cutlass_dtype.width,
    )

    cute.copy(copy_atom, thrA, thrFragA, pred=thrFragPred)
    cute.copy(copy_atom, thrB, thrFragB, pred=thrFragPred)

    res = op(thrFragA.load(), thrFragB.load())

    thrFragC.store(res)

    cute.copy(copy_atom, thrFragC, thrC, pred=thrFragPred)


@cute.jit
def elementwise(
    op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
    copy_bits: cutlass.Constexpr = 128,
):
    dtype = a.element_type
    vl_size = copy_bits // dtype.width

    # thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vl_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # Zipped devide -> ((TileM,TileN),(RestM,RestN))
    gA = cute.zipped_divide(a, tiler_mn)
    gB = cute.zipped_divide(b, tiler_mn)
    gC = cute.zipped_divide(c, tiler_mn)

    if VERBOSE:
        print(LOG, f"tiler_mn: {tiler_mn}")
        print(LOG, f"tv_layout: {tv_layout}")
        print(LOG, f"gA: {gA}")
        print(LOG, f"gB: {gB}")
        print(LOG, f"gC: {gC}")


    # Make a coord tensor
    Crd = cute.make_identity_tensor(c.shape)
    Crd = cute.zipped_divide(Crd, tiler_mn)

    grid_size = [cute.size(gC, mode=[1]), 1, 1]
    block_size = [cute.size(tv_layout, mode=[0]), 1, 1]

    elementwise_kernel(op, gA, gB, gC, Crd, c.shape, tv_layout).launch(
        grid=grid_size, block=block_size, stream=stream
    )


def run_elementwise(
    op,
    M,
    N,
    dtype: Type[cutlass.Numeric],
    is_a_dynamic_layout=False,
    is_b_dynamic_layout=False,
    is_c_dynamic_layout=False,
    skip_verify=False,
    warmup_iterations=10,
    iterations=100,
):
    print(f"Running elementwise {op.__name__} with M={M}, N={N}, dtype={dtype}")
    print(
        f"Dynamic layouts: A={is_a_dynamic_layout}, B={is_b_dynamic_layout}, C={is_c_dynamic_layout}"
    )

    torch_stream = torch.cuda.Stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # Allocate tensors with random values.
    torch_dtype = cutlass_torch.dtype(dtype)

    def _tensor_generator(return_torch=False):
        a = numpy.random.uniform(1, 100, size=(M, N))
        b = numpy.random.uniform(1, 100, size=(M, N))
        a = torch.from_numpy(a).to(device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.from_numpy(b).to(device=torch.device("cuda"), dtype=torch_dtype)
        c = torch.zeros_like(a)

        a_tensor = (
            from_dlpack(a) if not is_a_dynamic_layout else from_dlpack(a).mark_layout_dynamic()
        )
        b_tensor = (
            from_dlpack(b) if not is_b_dynamic_layout else from_dlpack(b).mark_layout_dynamic()
        )
        c_tensor = (
            from_dlpack(c) if not is_c_dynamic_layout else from_dlpack(c).mark_layout_dynamic()
        )

        if return_torch:
            return a, b, c, a_tensor, b_tensor, c_tensor

        return a_tensor, b_tensor, c_tensor

    if not skip_verify:
        _a, _b, _c, _a_tensor, _b_tensor, _c_tensor = _tensor_generator(return_torch=True)
        elementwise(op, _a_tensor, _b_tensor, _c_tensor, current_stream)
        torch.testing.assert_close(op(_a, _b), _c)
        print("Verification passed!")
    else:
        print("Verification skipped")

    # Compile
    compile_tic = time.perf_counter()
    elementwise_func = cute.compile(elementwise, op, *_tensor_generator(), current_stream)
    print(f"Kernel compiled in {time.perf_counter() - compile_tic:.4f} seconds")

    # Benchmarking
    average_kernel_time_us = testing.benchmark(
        elementwise_func,
        workspace_generator=lambda: testing.JitArguments(*_tensor_generator(), current_stream),
        workspace_count=10,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        stream=current_stream,
        use_cuda_graphs=True,
    )

    print(f"Kernel execution time: {average_kernel_time_us / 1e3:.4f} ms")
    print(
        f"Achieved memory throughput: {(3 * M * N * dtype.width / 8) / (average_kernel_time_us / 1e6) / 1e9:.2f} GB/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise ops to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--op", default="add", type=str, choices=operator.__all__)
    parser.add_argument("--M", "-M", default=1024, type=int)
    parser.add_argument("--N", "-N", default=1024, type=int)
    parser.add_argument("--dtype", default="float32", type=str)
    parser.add_argument("--warmup-iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    check_cuda()

    VERBOSE = args.verbose

    run_elementwise(
        getattr(operator, args.op),
        args.M,
        args.N,
        dtype=get_cutlass_dtype(args.dtype),
        is_a_dynamic_layout=False,
        is_b_dynamic_layout=False,
        is_c_dynamic_layout=False,
        skip_verify=args.skip_verify,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    print("PASS!")
