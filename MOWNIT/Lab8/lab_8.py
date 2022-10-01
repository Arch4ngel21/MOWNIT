# jeżeli promień spektralny < 1, to z każdego wektora początkowego rozwiązanie będzie zbiegać

import numpy as np
import random
from numpy.linalg import inv
import matplotlib.pyplot as plt


k = 11
m = 2
stop_mode = 0
stop_sigma = 1e-12


def a(i, j):
    global k, m
    if i == j:
        return k

    elif j == i - 1:
        return m / i

    elif j > i:
        return (-1)**j * m / j

    elif j < i - 1:
        return 0

    else:
        print(f"Blad przy tworzeniu macierzy, (i, j): {(i, j)}")


def stop_criteria(x_new, x_old, A, b):
    global stop_mode, stop_sigma

    if x_old is None:
        return False

    if stop_mode == 0:
        return np.sum(np.abs(x_new - x_old)) < stop_sigma
    elif stop_mode == 1:
        return np.sum(np.abs(A @ x_new - b)) < stop_sigma


def make_matrix(n: int, elem_type: np.dtype):
    A_tab = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            A_tab[i][j] = a(i+1, j+1)

    return np.array(A_tab, dtype=elem_type)


def jacoby_method(A, b, starting_vector, elem_type: np.dtype):
    steps = 0
    diagonal = np.diag(np.diag(A))
    upper = np.triu(A) - diagonal
    bottom = np.tril(A) - diagonal

    diagonal_inv = inv(diagonal)

    # x_approx = np.array([0 for _ in range(A.shape[0])])
    x_approx = np.array(starting_vector, dtype=elem_type)
    x_approx_old = None

    M = -diagonal_inv @ (bottom + upper)

    print(f"Promień spektralny macierzy: {np.max(np.linalg.eigvals(M))}")

    while not stop_criteria(x_approx, x_approx_old, A, b) and steps != 1000:
        x_approx_old = np.copy(x_approx)
        x_approx = M @ x_approx + diagonal_inv @ b
        steps += 1

    # print(f"x_approx:{x_approx}")
    return x_approx, steps


def compute_errors(x_real: np.ndarray, x_approx: np.ndarray):

    x_diff = np.absolute(x_real - x_approx)
    x_max = np.amax(x_diff)
    x_sqr = np.sum(np.power(x_diff, 2))

    return x_max, x_sqr


def test_1(n: int, elem_type: np.dtype):
    A = make_matrix(n, elem_type)

    rand_x = [-1, 1]

    x_real = [0 for _ in range(n)]

    for i in range(n):
        x_real[i] = rand_x[random.randint(0, 1)]

    x_real = np.array(x_real, dtype=elem_type)

    print(f"x_real: {x_real}")

    b = A @ x_real

    starting_vector = [random.randint(-100, 100) for _ in range(n)]
    print(f"starting_vector: {starting_vector}")

    x_approx, steps = jacoby_method(A, b, starting_vector, elem_type)
    x_max, x_sqr = compute_errors(x_real, x_approx)

    print(f"Wyliczony wektor x: {x_approx}")
    print(f"Max_diff error = {x_max}, sqr_diff error = {x_sqr}, steps = {steps}")
    print("\n")

    return steps, x_max, x_sqr


def tests(elem_type: np.dtype):
    n_s = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 175, 200]
    # n_s = range(3, 201)
    steps, x_max_s, x_sqr_s = np.array([], dtype=elem_type), np.array([], dtype=elem_type), np.array([], dtype=elem_type)

    for n in n_s:
        step, x_max, x_sqr = test_1(n, elem_type)
        steps = np.append(steps, step)
        x_max_s = np.append(x_max_s, x_max)
        x_sqr_s = np.append(x_sqr_s, x_sqr)

    return steps, x_max_s, x_sqr_s


if __name__ == '__main__':
    test_1(10, np.float32)
    test_1(5, np.float64)

    stop_mode = 1       # ustawienie kryterium stopu na normę z (Ax - b)

    test_1(7, np.float64)
