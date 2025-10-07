""" TSONG. 2025.06
Naive REDUCE F32 and F116 using CUTE
Key Notes:
    1. copy_bits = 128. 128bit per load/store instruction
"""
import argparse
import time
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

# Glue between torch and cutlass
import torch
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda


@cute.kernel
def reduce_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,
    shape: cute.Shape,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    # slice fro CTAs. ligical id --> addr
    blk_coord = ((None, None), bid)
    blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]

    pass


@cute.jit
def reduce(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
