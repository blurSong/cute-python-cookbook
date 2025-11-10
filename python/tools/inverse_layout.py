import cutlass
import cutlass.cute as cute


@cute.jit
def test():
    TV = cute.make_layout(((4, 8), (2, 2)), stride=((32, 1), (16, 8)))  # (T, V) -> (M, K)
    cute.printf("TV: {}\n", TV)
    A = cute.right_inverse(TV)  # (M, K) -> (T, V)
    cute.printf("A: {}\n", A)


if __name__ == "__main__":
    test()
