
import numpy as np
import matplotlib.pyplot as plt
from trigonometric_approximation import *


# Wartości stałych w zadanej funkcji f(x)
h = 1
k = 2
range_start = (-3)*np.pi
range_end = 3*np.pi


# Zadana funkcja f(x)
def f(x):
    global h, k
    return 10*h + (x**2)/k - 10*h*np.cos(k*x)


# Funkcja generująca węzły Czebyszewa
def chebyshev_points(a, b, n: int):
    x = np.arange(n, dtype=np.double)
    return np.double(0.5) * (a + b) + np.double(0.5) * (b - a) * np.cos((2*x + 1) * np.pi / (2*n))


# Pomocnicza funkcja do obliczania błędów
def compute_error_of_approximation(Y_base, Y_res):
    sup_abs_diff = np.max(np.abs(Y_base - Y_res))
    sqr_sum_diff = np.sum(np.power(Y_base - Y_res, 2))

    print(f"Największy błąd bezwzględny: {sup_abs_diff}")
    print(f"Suma kwadratów różnic: {sqr_sum_diff}")


def make_table_for_errors(m_s, n_s):
    sub_abs_diff_table = [['-' for _ in range(len(n_s))] for _ in range(len(m_s))]
    sqr_sum_diff_table = [['-' for _ in range(len(n_s))] for _ in range(len(m_s))]
    X_res = np.linspace(range_start, range_end, 200)
    Y_base = f(X_res)

    for e1, n_ in enumerate(n_s):

        for e2, m_ in enumerate(m_s):
            max_m_ = int((n_-1) // 2)
            if m_ > max_m_:
                continue
            X_ = np.linspace(range_start, range_end, n_)
            Y_ = f(X_)
            W_ = [1 for _ in range(n_)]

            solver_ = TrigonometricApproximation(X_, Y_, W_, m_)
            solver_.solve()

            Y_res = solver_.calc(X_res)
            sub_abs_diff_table[e2][e1] = round(np.max(np.abs(Y_base - Y_res)), 3)
            sqr_sum_diff_table[e2][e1] = round(np.sum(np.power(Y_base - Y_res, 2)), 3)

    print("sub_abs_diff:\n")
    for lane in sub_abs_diff_table:
        print(lane)

    print("\n\nsqr_sum_diff:\n")
    for lane in sqr_sum_diff_table:
        print(lane)


# Rysowanie błędów interpolacji dla zadanego stopnia i warunku brzegowego
def plot_error_of_an_approximation(n, m, mode):
    X_base = np.linspace(range_start, range_end, 500)
    Y_base = f(X_base)

    # Testowe punkty dla interpolacji
    if mode == 0:
        X_f = np.linspace(range_start, range_end, n)
        mode_text = " | Regular Points"
    else:
        X_f = chebyshev_points(range_start, range_end, n)
        mode_text = " | Chebyshev Points"
    Y_f = f(X_f)

    W = [1 for _ in range(n)]

    solver = TrigonometricApproximation(X_f, Y_f, W, m)

    solver.solve()
    X_res = np.linspace(range_start, range_end, 500)
    Y_res = solver.calc(X_res)

    Y_error = np.abs(Y_res - Y_base)
    plt.scatter(X_base, Y_error, c='purple', s=4)
    plt.title("Error for Polynomial Approximation" + mode_text)
    plt.show()


# Do rysowania interpolowanej funkcji
def plot_approximating_function():
    X_base = np.linspace(range_start, range_end, 200)
    Y_base = f(X_base)
    plt.plot(X_base, Y_base)
    plt.title("Aproksymowana funkcja")
    plt.grid(visible=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_approx(n, m, mode):
    X_base = np.linspace(range_start, range_end, 200)
    Y_base = f(X_base)
    if mode == 0:
        X_for_approx = np.linspace(range_start, range_end, n)
        mode_text = " | Regular Points"
    else:
        X_for_approx = chebyshev_points(range_start, range_end, n)
        mode_text = " | Chebyshev Points"
    Y_for_approx = f(X_for_approx)
    W = [1 for _ in range(n)]

    solver = TrigonometricApproximation(X_for_approx, Y_for_approx, W, m)
    solver.solve()

    Y_approx = solver.calc(X_base)
    plt.plot(X_base, Y_base, label="Approximated function")
    plt.scatter(X_for_approx, Y_for_approx, c='red', label="Points")
    plt.plot(X_base, Y_approx, c='green', label="Approximating function")
    plt.title(f"Trigonometric Approx, m = {m}, n = {n}" + mode_text)
    plt.legend(loc="lower left")


if __name__ == '__main__':
    m_s = [1, 2, 3, 5, 7, 10, 12, 15, 30, 50]
    n_s = [5, 7, 10, 12, 15, 25, 30, 50, 70, 100, 120, 150, 200, 300, 400, 500, 600, 800]

    plot_approximating_function()   # wykres aproksymowanej funkcji
    mode = 0    # 0 - równoodległe punkty   # 1 - węzły Czebyszewa

    # Testy dla zdefiniowanych wcześniej wartości n i m
    for n in n_s:
        for m in m_s:
            max_m = int((n-1) // 2)
            if m > max_m:
                continue
            plot_approx(n, m, mode)
            plt.show()


    # Wypisanie tabeli błędów aproksymacji
    make_table_for_errors(m_s, n_s)

    # Funkcja rysująca różnice pomiędzy funkcją aproksymowaną, a funkcją aproksymującą
    plot_error_of_an_approximation(100, 4, mode)
