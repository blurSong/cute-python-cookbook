"""TSONG. 2025.06
A playgroud for CUTE layouts helping to understand.
https://docs.nvidia.com/cutlass/media/docs/cpp/cute/01_layout.html
https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.make_layout

Cutlass unit tests: https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core

About create dynamic layouts: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

# Glue between torch and cutlass

import cuda.bindings.driver as cuda


@cute.jit
def make_layouts():
    print("\n[Make Layouts]")
    # Basic layouts
    layout_4x16_rowmaj = cute.make_layout((4, 16), stride=(16, 1))
    layout_4x16_colmaj = cute.make_layout((4, 16))
    layout_4x2_8_rowmaj = cute.make_layout((4, (2, 8)), stride=(16, (8, 1)))
    layout_4x2_8_colmaj = cute.make_layout((4, (2, 8)))
    layout_4x16x32_rowmaj = cute.make_layout((4, 16, 32), stride=(512, 32, 1))
    layout_4x16x32_colmaj = cute.make_layout((4, 16, 32))
    layout_4x16x4_8_rowmaj = cute.make_layout((4, 16, (4, 8)), stride=(512, 32, (8, 1)))
    layout_4x16x4_8_colmaj = cute.make_layout((4, 16, (4, 8)))

    print("\nBasic layouts:")
    print("Note that the cute default layout is left-most(column major).")
    print(f"2d layout 4x16 row major: {layout_4x16_rowmaj}")
    print(f"2d layout 4x16 column major: {layout_4x16_colmaj}")
    print(f"2d layout 4x(2x8) row major: {layout_4x2_8_rowmaj}")
    print(f"2d layout 4x(2x8) column major: {layout_4x2_8_colmaj}")
    print(f"3d layout 4x16x32 row major: {layout_4x16x32_rowmaj}")
    print(f"3d layout 4x16x32 column major: {layout_4x16x32_colmaj}")
    print(f"3d layout 4x16x(4x8) row major: {layout_4x16x4_8_rowmaj}")
    print(f"3d layout 4x16x(4x8) column major: {layout_4x16x4_8_colmaj}")

    # Identity layouts
    layout_4x16_ident = cute.make_identity_layout((4, 16))
    layout_4x16x32_ident = cute.make_identity_layout((4, 16, 32))
    print("\nIdentity layouts:")
    print(f"2d layout 4x16 identity: {layout_4x16_ident}")
    print(f"3d layout 4x16x32 identity: {layout_4x16x32_ident}")

    # Ordered layouts
    # The order parameter specifies the ordering of dimensions from fastest-varying to slowest-varying
    # The length of order matches the rank of the shape
    layout_4x16_ordered_10 = cute.make_ordered_layout((4, 16), order=(1, 0))  # row major
    layout_4x16_ordered_01 = cute.make_ordered_layout((4, 16), order=(0, 1))  # column major
    layout_4x16x32_ordered_201 = cute.make_ordered_layout((4, 16, 32), order=(2, 0, 1))
    layout_4x16x32_ordered_210 = cute.make_ordered_layout((4, 16, 32), order=(2, 1, 0))
    layout_4x16x4_8_ordered_210 = cute.make_ordered_layout((4, 16, (4, 8)), order=(2, 1, 0))
    print("\nOrdered layouts:")
    print(f"2d layout 4x16 ordered (1,0): {layout_4x16_ordered_10}")
    print(f"2d layout 4x16 ordered (0,1): {layout_4x16_ordered_01}")
    print(f"3d layout 4x16x32 ordered (2,0,1): {layout_4x16x32_ordered_201}")
    print(f"3d layout 4x16x32 ordered (2,1,0): {layout_4x16x32_ordered_210}")
    print(f"3d layout 4x16x(4x8) ordered (2,1,0): {layout_4x16x4_8_ordered_210}")

    # Composed layouts
    # Inner transformation can be a layout or swizzle
    # The composition applies transformations in the order: outer → offset → inner
    siwzzle = cute.make_swizzle(3, 3, 3)  # b, m, s
    layout_128x512 = cute.make_layout((128, 512))
    layout_128x512_swizzled = cute.make_composed_layout(siwzzle, 0, layout_128x512)
    print("\nComposed(Swizzle) layout:")
    print(f"layout 4x8: {layout_128x512}")
    print(f"swizzled layout 4x8: {layout_128x512_swizzled}")

    # TV layouts
    # Create a [tiled copy] given separate thr and val layouts.
    thr_layout = cute.make_ordered_layout((5, 8), order=(1, 0))
    val_layout = cute.make_ordered_layout((5, 64), order=(1, 0))
    tv_shape, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print("\nTV layout:")
    print(f"thr layout: {thr_layout}, val layout: {val_layout}")
    print(f"tv shape: {tv_shape}, tv layout: {tv_layout}")


@cute.jit
def layout_manipulation():
    print("\n[Layout Manipulation]")

    yjsp_layout = cute.make_layout((114, 514, 1919, 810))
    yjsp_idx1 = cute.crd2idx((1, 1, 1, 1), yjsp_layout)
    yjsp_idx2 = cute.crd2idx((1, 2, 3, 4), yjsp_layout)
    yjsp_idx3 = cute.crd2idx((10, 10, 10, 10), yjsp_layout)
    # yjsp_coord1 = cute.idx2crd(yjsp_idx1, yjsp_layout)
    # yjsp_coord2 = cute.idx2crd(yjsp_idx2, yjsp_layout)
    # yjsp_coord3 = cute.idx2crd(yjsp_idx3, yjsp_layout)
    print(f"\nYJSP layout: {yjsp_layout}")
    print(f"Index for (1, 1, 1, 1): {yjsp_idx1}")
    print(f"Index for (1, 2, 3, 4): {yjsp_idx2}")
    print(f"Index for (10, 10, 10, 10): {yjsp_idx3}")
    # print(f"Coordinate for index {yjsp_idx1}: {yjsp_coord1}")
    # print(f"Coordinate for index {yjsp_idx2}: {yjsp_coord2}")
    # print(f"Coordinate for index {yjsp_idx3}: {yjsp_coord3}")

    yjsp_layout_01 = cute.select(yjsp_layout, (0, 1))
    yjsp_layout_23 = cute.select(yjsp_layout, (2, 3))
    print("\nSelected layouts:")
    print(f"Selected layout (0, 1): {yjsp_layout_01}")
    print(f"Selected layout (2, 3): {yjsp_layout_23}")

    # yjsp_layout_0123 = cute.make_layout(yjsp_layout_01, yjsp_layout_23)
    # yjsp_layout_012301 = cute.make_layout(yjsp_layout_01, yjsp_layout_23, yjsp_layout_01)
    # print("\nConcatenated layouts:")
    # print(f"Concatenated layout (0, 1, 2, 3): {yjsp_layout_0123}")
    # print(f"Concatenated layout (0, 1, 2, 3, 0, 1): {yjsp_layout_012301}")

    """NOTE Cute dls has a bug here.
    core.py:2219 --> return _cute_ir.group_modes(input, begin, end, loc=loc, ip=ip)
    """
    # end= is exclusive
    yjsp_layout_grouped_13 = cute.group_modes(yjsp_layout, 1, 3)
    yjsp_layout_grouped_24 = cute.group_modes(yjsp_layout, 2, 4)
    print("\nGrouped layouts:")
    print(f"Grouped layout (1, 3): {yjsp_layout_grouped_13}")
    print(f"Grouped layout (2, 4): {yjsp_layout_grouped_24}")

    """NOTE Not supported yet.
    """
    # yjsp_layout_flatt = cute.flatten(yjsp_layout)
    # print("\nFlattened layout:")
    # print(f"Flattened layout: {yjsp_layout_flatt}")


@cute.jit
def layout_algebra():
    print("\n[Layout Algebra]")

    # Coalesce
    """The coalesce operation is a "simplify" of the layout.
    Not all layouts can be coalesced, they need to be "coalescable".
    One is, s0:d0  ++  s1:s0*d0  =>  s0*s1:d0.
    --> If the second mode’s stride is the product of the first mode’s size and stride, then they can be combined.

    By-mode Coalesce
    Use step to identify the mode by 1 after coalesced.

    https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html#coalesce
    """
    layout_example = cute.make_layout((4, (2, 5)), stride=(1, (5, 2)))
    layout_coalesced = cute.coalesce(layout_example)
    layout_coalesced2 = cute.coalesce(layout_example, target_profile=(1, 1))
    print("\nCoalesce:")
    print(f"Original layout: {layout_example}")
    print(f"Coalesced layout: {layout_coalesced}")
    print(f"Coalesced layout with target profile (1, 1): {layout_coalesced2}")

    # Composition
    """The composition is the core of CuTe and is used in just about every higher-level operation
    https://github.com/NVIDIA/cutlass/blob/main/test/unit/cute/core/composition.cpp

    > Compatible condition

    > Functional composition R := A o B.  R(c) := (A o B)(c) := A(B(c))
        One word: A o B = A o s:d, for integral s and d means that we want
            (1) "remove" the first d elements from A (computed by progressively "dividing out" \
                the first d elements from the shape of A starting from the left.)
            (2) then "keep" the first s of those strided elements (computed by "modding out" \
                the first s elements from the shape of A starting from the left).
            Finally, let all the untouched shapes to be 1:

        E.G. (10,2):(16,4) o (5,4):(1,5)
            1. View B as sublayouts (2 1-D modes)
                := (10,2):(16,4) o (5:1, 4:5)
            2. Left distributivity.
                (10,2):(16,4) o (5:1), (10,2):(16,4) o (4:5)
                (10,2):(16,4) o (5:1) = (10/1%5,:2->1):(16*1,4) = (5,1):(16,4) = 5:16   # strip the mode with shape=1
                (10,2):(16,4) o (4:5) = (10/5%4,2%(4/2)):(16*5,4) = (2,2):(80,4)
            3. Compose layouts and By-mode coalescing
                (5:16, (2,2):(80,4) = (5,(2,2)):(16,(80,4)) = (5,(2,2)):(16,(80,4))

    > By-mode Composition: use multiple tilers to compose a layout by modes.

        E.G. (12,(4,8)):(59,(13,1)) o (3:4,8:2) By-mode

            = (12:59 o 3:4,(4,8):(13,1) o 8:2)
            = (3:236,(2,4):(26,1))
            = (3,(2:4)):(236,(26,1))
    """
    layout_a = cute.make_layout((10, 2), stride=(16, 4))
    layout_b = cute.make_layout((5, 4), stride=(1, 5))
    layout_c = cute.composition(layout_a, layout_b)
    print("\nComposition:")
    print(f"Layout A: {layout_a}")
    print(f"Layout B: {layout_b}")
    print(f"Layout C := A o B: {layout_c}")
    layout_a2 = cute.make_layout((12, (4, 8)), stride=(59, (13, 1)))
    tiler_b = (cute.make_layout(3, stride=4), cute.make_layout(8, stride=2))
    layout_c2 = cute.composition(layout_a2, tiler_b)
    # layout_c2_same = cute.make_layout(
    #     cute.composition(layout_a2[0], tiler_b[0]), cute.composition(layout_a2[1], tiler_b[1]))
    print("\nBy-mode Composition:")
    print(f"Layout A: {layout_a2}")
    print(f"Tiler B: {tiler_b}")
    print(f"Layout C := A o B: {layout_c2}")
    # print(f"Layout C same := A o B: {layout_c2_same}")

    # Complement
    """ The complement of a layout B attempts to find another layout that \
    represents the "rest elements" that aren't touched by the layout A.
    Whereas, the cotarget param indicates the codomain size of layout A.

    E.G. complement(4:2, 24) is (2,3):(1,8).
        Note that (4,(2,3)):(2,(1,8)) has cosize 24.
        The "hole" in 4:2 is filled with 2:1 first, then everything is repeated 3 times with 3:8.
    """
    print(f"complement(4:1, 24) is {cute.complement(cute.make_layout(4, stride=1), 24)}")
    print(f"complement(6:1, 24) is {cute.complement(cute.make_layout(6, stride=1), 24)}")
    print(f"complement((4,6):(1,4), 24) is {cute.complement(cute.make_layout((4, 6), stride=(1, 4)), 24)}")
    print(f"complement(4:2, 24) is {cute.complement(cute.make_layout(4, stride=2), 24)}")
    print(f"complement((2,4):(1,6), 24) is {cute.complement(cute.make_layout((2, 4), stride=(1, 6)), 24)}")
    print(f"complement((2,2):(1,6), 24) is {cute.complement(cute.make_layout((2, 2), stride=(1, 6)), 24)}")

    # Division
    """ Functional A ⊘ B := A o (B, B*)
        logical_divide(A, B) = composition(A, make_layout(B, complement(B, size(A))))
        layout (B, B*) is the concantenation of B and B*.
        Mode-1 are all A elements pointed to by B.
        Mode-2 are all size(A) elements not pointed to by B, i.e., B*.
    Examples & vision https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html#logical-divide-1-d-example

    > Zipped,Tiled,Flat Divides
    These divide functions are kinds of "permutation" of the logical divide.
    ------------------------------------------------------------------
    |    Layout Shape : (M, N, L, ...)                               |
    |    Tiler Shape  : <TileM, TileN>                               |
    |                                                                |
    |    logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)     |
    |    zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...  ))     |
    |    tiled_divide   : ((TileM,TileN), RestM, RestN, L, ... )     |
    |    flat_divide    : (TileM, TileN, RestM, RestN, L, ... )      |
    ------------------------------------------------------------------
    """
    layout_a = cute.make_layout((4, 2, 3), stride=(2, 1, 8))
    tiler_b = cute.make_layout(4, stride=2)
    layout_a_logdiv_b = cute.logical_divide(layout_a, tiler_b)
    # layout_a_logdiv_b_same = cute.composition(
    #     layout_a, cute.make_layout(tiler_b, cute.complement(tiler_b, size=cute.size(layout_a))))

    layout_a2 = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))
    tiler_b2_mod1 = cute.make_layout(3, stride=3)
    tiler_b2_mod2 = cute.make_layout((2, 4), stride=(1, 8))
    layout_a2_logdiv_b2 = cute.logical_divide(layout_a2, (tiler_b2_mod1, tiler_b2_mod2))

    print("\nLogical Division:")
    print(f"Layout A: {layout_a}")
    print(f"Tiler B: {tiler_b}")
    print(f"Layout A ⊘ B: {layout_a_logdiv_b}")
    print("\nLogical Division By-mode:")
    print(f"Layout A2: {layout_a2}")
    print(f"Tiler B2: {tiler_b2_mod1} and {tiler_b2_mod2}")
    print(f"Layout A2 ⊘ B2: {layout_a2_logdiv_b2}")

    # Product
    """ Functional A ⨂ B := (A, A* o B)
        The first mode is just the layout A.
        The second mode is the layout B but with each element replaced by a "unique replication" of layout A.
        And A* is the complement of A, up to the size(A)*cosize(B).

    > Blocked and Raked Products
    By-mode product is not recommended, because it is not intuitive to express
    "Tile A according to B1 and B2 across the row/col modes."
    You can see below we use 2 weird tiler strides(why 5 and 6?). They are computed based on A and the production layout we want!

    Blocked/Ranked products are rank-sensitive transformations on top of 1-D logical_product
    to offer more sensible functions to replace the confusing by-mode product.

    (xa, ya) ⨂ (xb, yb)
    -----------------------------
    logical_	((xa,ya),(xb,yb))
    blocked_	((xa,xb),(ya,yb))
    raked_	    ((xb,xa),(yb,ya))

    > Zipped, Tiled and Flat
    Same as devides. Just simply rearrange the modes that result from a by-mode logical_product.
    """
    layout_a = cute.make_layout((2, 2), stride=(4, 1))
    tiler_b = cute.make_layout(6, stride=1)
    layout_a_prod_b = cute.logical_product(layout_a, tiler_b)
    layout_a_bymode = cute.make_layout((2, 5), stride=(5, 1))
    tiler_b_bymode_1 = cute.make_layout(3, stride=5)
    tiler_b_bymode_2 = cute.make_layout(4, stride=6)
    # layout_a_prod_b_bymode = cute.logical_product(layout_a_bymode, (tiler_b_bymode_1, tiler_b_bymode_2))
    layout_b_bymode = cute.make_layout((3, 4), stride=(1, 3))
    layout_a_prod_b_blocked = cute.blocked_product(layout_a_bymode, layout_b_bymode)
    layout_a_prod_b_raked = cute.raked_product(layout_a_bymode, layout_b_bymode)
    print("\nLogical Product:")
    print(f"Layout A: {layout_a}")
    print(f"Tiler B: {tiler_b}")
    print(f"Layout A ⨂ B: {layout_a_prod_b}")
    print("\nLogical Product By-mode ")
    print(f"Layout A: {layout_a_bymode}")
    print(f"Tiler B: {tiler_b_bymode_1} and {tiler_b_bymode_2}. Wired!")
    # print(f"Layout A2 ⨂ B2: {layout_a_prod_b_bymode}")
    print("\nBlocked Product:")
    print(f"Layout A: {layout_a_bymode}")
    print(f"Layout B: {layout_b_bymode}. Nice!")
    print(f"Layout A ⨂ B Blocked: {layout_a_prod_b_blocked}")
    print("\nRaked Product:")
    print(f"Layout A: {layout_a_bymode}")
    print(f"Layout B: {layout_b_bymode}. Nice!")
    print(f"Layout A ⨂ B Raked: {layout_a_prod_b_raked}")


make_layouts_func = cute.compile(make_layouts)
layout_manipulation_func = cute.compile(layout_manipulation)
layout_algebra_func = cute.compile(layout_algebra)

make_layouts_func()
layout_manipulation_func()
layout_algebra_func()
