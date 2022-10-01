
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv


k = 5
m = 3


def a_1(i, j) -> int:
    if i == 1:
        return 1

    return 1 / (i + j - 1)


def a_2(i, j) -> int:
    if j >= i:
        return 2*i / j
    else:
        return a_2(j, i)


def make_vector_x(n: int, elem_type: np.dtype) -> np.ndarray:
    new_x = []
    values = (-1, 1)
    for _ in range(n):
        index = random.randint(0, 1)
        new_x.append(values[index])

    return np.array(new_x, dtype=elem_type)


def _conditioning(matrix):
    C = np.sum(matrix, axis=1)

    val_1 = np.max(C)
    matrix_inv = inv(matrix)

    C = np.sum(matrix_inv, axis=1)
    val_2 = np.max(C)

    return val_1 * val_2


def conditioning(n: int, elem_type: np.dtype):
    print(f"Uwarunkowanie dla macierzy dla n = {n}:")

    matrix_1 = make_matrix(n, elem_type, 0)
    matrix_2 = make_matrix(n, elem_type, 1)

    res_1 = _conditioning(matrix_1)
    res_2 = _conditioning(matrix_2)

    print(f"Macierz z zad. 1: {res_1}, Macierz z zad. 2: {res_2}")

    if res_1 == res_2:
        print("Macierze są identycznie uwarunkowane")
    elif res_1 < res_2:
        print("Macierz z zad. 1 jest lepiej uwarunkowana")
    else:
        print("Macierz z zad. 2 jest lepiej uwarunkowana")

    return res_1, res_2


def test_conditioning(elem_type: np.dtype):
    n_s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 175, 200]

    res_file = open("res_cond.txt", "w")
    for n in n_s:
        res_1, res_2 = conditioning(n, elem_type)
        res_file.write(str(n) + " " + str(res_1) + " " + str(res_2) + "\n")

    res_file.close()


def make_matrix(n: int, elem_type: np.dtype, mode=0) -> np.ndarray:
    M = [[0 for _ in range(n)] for _ in range(n)]

    for y in range(n):
        for x in range(n):
            if mode == 0:
                M[y][x] = a_1(x+1, y+1)
            else:
                M[y][x] = a_2(x+1, y+1)

    return np.array(M, dtype=elem_type)


# O(n^3) - metoda wstecznego podstawiania
def solve_gauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = b.shape[0]

    for k in range(n-1):
        for i in range(k+1, n):
            L = A[i, k] / A[k, k]
            for j in range(k+1, n):
                A[i, j] -= L * A[k, j]
            b[i] -= L * b[k]

    x = b.copy()
    for i in range(n-1, -1, -1):
        S = 0.0
        for j in range(i+1, n):
            S += A[i, j] * x[j]
        x[i] = (x[i] - S) / A[i, i]

    return x


def test(n: int, elem_type: np.dtype):
    x_real = make_vector_x(n, elem_type)
    A = make_matrix(n, elem_type, 0)

    b = A @ x_real

    print("A:\n", A)
    print("x rzeczywisty:\n", x_real)
    print("b:\n", b)

    x_approx = solve_gauss(A, b)
    print("x obliczony:\n", x_approx)


def time_tests_1(elem_type: np.dtype, mode=0):
    if mode == 0:
        print("Testy dla zadania 1")
    else:
        print("Testy dla zadania 2")
    n_s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 175, 200]
    # n_s = range(3, 201)

    # res_file = open("res_times_2_f_64.txt", "w")
    # res_file_err = open("res_err_2_f_64.txt", "w")

    time_points = [[], []]
    err_points_max = [[], []]
    err_points_sqr = [[], []]

    for n in n_s:
        x_real = make_vector_x(n, elem_type)
        A = make_matrix(n, elem_type, mode)

        b = A @ x_real

        start = time.time()
        x_approx = solve_gauss(A, b)
        # res_file.write(str(n) + " " + str(time.time() - start) + "\n")

        print(f"Time for {n}: {time.time() - start}")
        time_points[0].append(n)
        time_points[1].append(time.time() - start)

        x_max, x_sqr = compute_errors(x_real, x_approx)
        err_points_max[0].append(n)
        err_points_max[1].append(x_max)

        err_points_sqr[0].append(n)
        err_points_sqr[1].append(x_sqr)
        # res_file_err.write(str(n) + " " + str(x_max) + " " + str(x_sqr) + "\n")

    # res_file.close()
    # res_file_err.close()

    return time_points, err_points_max, err_points_sqr


def a_3(i, j) -> int:

    global k, m

    if j < i - 1 or j > i + 1:
        return 0

    elif j == i + 1:
        return 1 / (i + m)

    elif i > 1 and j == i - 1:
        return k / (i + m + 1)

    elif i == j:
        return k

    else:
        print(f"Błąd w liczeniu a!: {i, j}")


def make_matrix_3(n: int, elem_type: np.dtype) -> np.ndarray:

    M = [[0 for _ in range(n)] for _ in range(n)]

    for y in range(n):
        for x in range(n):
            M[y][x] = a_3(x+1, y+1)

    return np.array(M, dtype=elem_type)


def thomas_algorithm(A, d, elem_type):

    n = d.shape[0]
    a = np.array([0], dtype=elem_type)
    a = np.append(a, [A[i, i+1] for i in range(n-1)])

    b = np.array([], dtype=elem_type)
    b = np.append(b, [A[i, i] for i in range(n)])
    c = np.array([], dtype=elem_type)
    c = np.append(c, [A[i+1, i] for i in range(n-1)])
    c = np.append(c, [0])
    beta = np.array([(-c[0]) / b[0]], dtype=elem_type)
    gamma = np.array([d[0] / b[0]], dtype=elem_type)

    for i in range(1, n):
        beta = np.append(beta, -(c[i] / (a[0] * beta[i-1] + b[i])))
        gamma = np.append(gamma, (d[i] - a[i] * gamma[i-1]) / (a[i] * beta[i-1] + b[i]))

    x = [0 for _ in range(n)]
    x[n-1] = gamma[n-1]

    for i in range(n-2, -1, -1):
        x[i] = beta[i] * x[i+1] + gamma[i]

    return np.array(x)


def thomas_algorithm_v4(A: np.ndarray, d: np.ndarray, elem_type: np.dtype):
    n = d.shape[0]
    C = np.array([0 for _ in range(n)], dtype=elem_type)
    C[0] = A[0, 0]
    x = np.array([0 for _ in range(n)], dtype=elem_type)
    x[0] = d[0]

    for i in range(1, n):
        factor = A[i, i - 1] / C[i - 1]
        C[i] = A[i, i] - factor * A[i - 1, i]
        x[i] = d[i] - factor * x[i - 1]

    x[n - 1] = x[n - 1] / C[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (x[i] - A[i, i + 1] * x[i + 1]) / C[i]

    return x


def test_2(n: int, elem_type: np.dtype):
    x_real = make_vector_x(n, elem_type)
    A = make_matrix_3(n, elem_type)

    b = A @ x_real

    print("A:\n", A)
    print("x_real:\n", x_real)
    print("b:\n", b)

    x_approx = thomas_algorithm_v4(A, b, elem_type)
    print("x_approx:\n", x_approx)


def compute_errors(x_real: np.ndarray, x_approx: np.ndarray):

    x_diff = np.absolute(x_real - x_approx)
    x_max = np.amax(x_diff)
    x_sqr = np.sum(np.power(x_diff, 2))

    return x_max, x_sqr


def time_tests_2(elem_type: np.dtype):
    # n_s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 175, 200]
    n_s = range(3, 201)

    time_points_1 = [[], []]
    err_points_max_1 = [[], []]
    err_points_sqr_1 = [[], []]
    time_points_2 = [[], []]
    err_points_max_2 = [[], []]
    err_points_sqr_2 = [[], []]

    # res_file = open("res_times_3_f_64.txt", "w")
    # res_file_err = open("res_err_3_f_64.txt", "w")

    for n in n_s:
        x_real = make_vector_x(n, elem_type)
        A = make_matrix_3(n, elem_type)
        b = A @ x_real

        start = time.time()
        x_approx_1 = thomas_algorithm_v4(A, b, elem_type)

        # res_file.write(str(n) + " " + str(time.time() - start) + " ")
        x_max, x_sqr = compute_errors(x_real, x_approx_1)
        # res_file_err.write(str(n) + " " + str(x_max) + " " + str(x_sqr) + " ")
        time_points_1[0].append(n)
        time_points_1[1].append(time.time() - start)
        err_points_max_1[0].append(n)
        err_points_max_1[1].append(x_max)
        err_points_sqr_1[0].append(n)
        err_points_sqr_1[1].append(x_sqr)
        print(f"Time for {n}: Thomas: {time.time() - start}", end=" ")

        start = time.time()
        x_approx_2 = solve_gauss(A, b)

        # res_file.write(str(time.time() - start) + "\n")
        x_max, x_sqr = compute_errors(x_real, x_approx_2)
        time_points_2[0].append(n)
        time_points_2[1].append(time.time() - start)
        err_points_max_2[0].append(n)
        err_points_max_2[1].append(x_max)
        err_points_sqr_2[0].append(n)
        err_points_sqr_2[1].append(x_sqr)
        # res_file_err.write(" " + str(x_max) + " " + str(x_sqr) + "\n")
        print(f"Gauss: {time.time() - start}")

    # res_file.close()
    # res_file_err.close()

    return time_points_1, err_points_max_1, err_points_sqr_1, time_points_2, err_points_max_2, err_points_sqr_2


if __name__ == '__main__':

    conditioning(20, np.float64)
    print("\nPrzykładowy test dla zad.1:")
    test(5, np.float32)

    print("\n")
    time_tests_1(np.float32, 0)
    print("\n")
    time_tests_1(np.float64, 1)
    print("\n")

    plt.legend(loc="lower left")

    t_1, e_m_1, e_s_1, t_2, e_m_2, e_s_2 = time_tests_2(np.float32)
    t_3, e_m_3, e_s_3, t_4, e_m_4, e_s_4 = time_tests_2(np.float64)

    plt.scatter(t_1[0], t_1[1], c='green', label="Thomas float32")
    plt.scatter(t_2[0], t_2[1], c='blue', label="Gauss float32")
    plt.scatter(t_3[0], t_3[1], c='red', label="Thomas float64")
    plt.scatter(t_4[0], t_4[1], c='orange', label="Gauss float64")
    plt.legend(loc="lower left")
    plt.title("Times comparison for float32 and float64 for exercise 3")
    plt.show()

    plt.scatter(e_m_1[0], e_m_1[1], c='green', label="Thomas float32")
    plt.scatter(e_m_2[0], e_m_2[1], c='blue', label="Gauss float32")
    plt.scatter(e_m_3[0], e_m_3[1], c='red', label="Thomas float64")
    plt.scatter(e_m_4[0], e_m_4[1], c='orange', label="Gauss float64")
    plt.legend(loc="lower left")
    plt.title("Max error comparison for float32 and float64 for exercise 3")
    plt.show()

    plt.scatter(e_s_1[0], e_s_1[1], c='green', label="Thomas float32")
    plt.scatter(e_s_2[0], e_s_2[1], c='blue', label="Gauss float32")
    plt.scatter(e_s_3[0], e_s_3[1], c='red', label="Thomas float64")
    plt.scatter(e_s_4[0], e_s_4[1], c='orange', label="Gauss float64")
    plt.legend(loc="lower left")
    plt.title("Square difference error comparison for float32 and float64 for exercise 3")
    plt.show()
