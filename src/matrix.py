import numpy as np
import scipy.linalg
from numpy.typing import NDArray


def make_identity_homography() -> NDArray:
    H = np.eye(3)
    return H


def make_translation_homography(dx: float | int, dy: float | int) -> NDArray:
    H = make_identity_homography()
    H[0, 2] = dx
    H[1, 2] = dy
    return H


def make_matrix(rows: int, cols: int) -> NDArray:
    return np.zeros((rows, cols))


def copy_matrix(m: NDArray) -> NDArray:
    return np.copy(m)


def augment_matrix(m: NDArray) -> NDArray:
    rows, cols = m.shape
    I: NDArray = np.eye(rows)
    augmented: NDArray = np.hstack((m, I))
    return augmented


def make_identity(rows: int, cols: int) -> NDArray:
    assert rows == cols, "Identity matrix must be square"
    return np.eye(rows)


def matrix_mult_matrix(a: NDArray, b: NDArray) -> NDArray:
    return np.dot(a, b)


def matrix_sub_matrix(a: NDArray, b: NDArray) -> NDArray:
    return np.subtract(a, b)


def transpose_matrix(m: NDArray) -> NDArray:
    return np.transpose(m)


def scale_matrix(m: NDArray, s: int) -> None:
    m *= s


def matrix_mult_vector(m:NDArray , v: NDArray) -> NDArray:
    return np.dot(m, v)


def print_matrix(m: NDArray) -> None:
    print(m)


def LUP_solve(L, U, p, b):
    # Using NumPy solve as a stand-in for LUP_solve functionality
    return np.linalg.solve(L @ U, b[p])


def matrix_invert(m: NDArray) -> NDArray:
    return np.linalg.inv(m)


def in_place_LUP(m):
    P, L, U = scipy.linalg.lu(m)
    return L, U, P


def random_matrix(rows: int, cols: int) -> NDArray:
    return np.random.rand(rows, cols) * 100 - 50


def sle_solve(A: NDArray, b: NDArray) -> NDArray:
    return np.linalg.solve(A, b)


def solve_system(M: NDArray, b: NDArray):
    Mt = transpose_matrix(M)
    MtM = matrix_mult_matrix(Mt, M)
    MtMinv = matrix_invert(MtM)
    Mdag = matrix_mult_matrix(MtMinv, Mt)
    a = matrix_mult_matrix(Mdag, b)
    return a
