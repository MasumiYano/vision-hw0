import numpy as np
import scipy.linalg


def make_identity_homography():
    H = np.eye(3)
    return H


def make_translation_homography(dx, dy):
    H = make_identity_homography()
    H[0, 2] = dx
    H[1, 2] = dy
    return H


def make_matrix(rows, cols):
    return np.zeros((rows, cols))


def copy_matrix(m):
    return np.copy(m)


def augment_matrix(m):
    rows, cols = m.shape
    I = np.eye(rows)
    augmented = np.hstack((m, I))
    return augmented


def make_identity(rows, cols):
    assert rows == cols, "Identity matrix must be square"
    return np.eye(rows)


def matrix_mult_matrix(a, b):
    return np.dot(a, b)


def matrix_sub_matrix(a, b):
    return np.subtract(a, b)


def transpose_matrix(m):
    return np.transpose(m)


def scale_matrix(m, s):
    m *= s


def matrix_mult_vector(m, v):
    return np.dot(m, v)


def print_matrix(m):
    print(m)


def LUP_solve(L, U, p, b):
    # Using NumPy solve as a stand-in for LUP_solve functionality
    return np.linalg.solve(L @ U, b[p])


def matrix_invert(m):
    return np.linalg.inv(m)


def in_place_LUP(m):
    P, L, U = scipy.linalg.lu(m)
    return L, U, P


def random_matrix(rows, cols):
    return np.random.rand(rows, cols) * 100 - 50


def sle_solve(A, b):
    return np.linalg.solve(A, b)


def solve_system(M, b):
    Mt = transpose_matrix(M)
    MtM = matrix_mult_matrix(Mt, M)
    MtMinv = matrix_invert(MtM)
    Mdag = matrix_mult_matrix(MtMinv, Mt)
    a = matrix_mult_matrix(Mdag, b)
    return a
