import re

def swizzle(offset, b_bits, m_base, s_shift=None):
    """
    CuTe like swizzle function.
    https://github.com/NVIDIA/cutlass/include/cute/swizzle.hpp#L55
    Args:
        offset (int): The offset to swizzle.
        b_bits (int): The number of bits in the base address.
        m_base (int): The base address for the swizzle.
        s_shift (int): The shift value for the swizzle.
    """
    if s_shift is None:
        s_shift = b_bits
    assert b_bits >= 0 and m_base >= 0 and abs(s_shift) >= b_bits

    bit_mask = (1 << b_bits) - 1
    yyy_mask = bit_mask << (m_base + max(0, s_shift))
    zzz_mask = bit_mask << (m_base - min(0, s_shift))
    mask_shift = s_shift

    swizzle_code = yyy_mask | zzz_mask
