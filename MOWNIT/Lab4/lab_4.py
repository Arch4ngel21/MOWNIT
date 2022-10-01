# Michał Szafarczyk

import numpy as np
import matplotlib.pyplot as plt

from quadratic_spline import *
from cubic_spline import *


# Wartości stałych w zadanej funkcji f(x)
m = 1
k = 2
range_start = (-3)*np.pi
range_end = 3*np.pi

# Ilość punktów dla obliczania błędów
error_points = 500

# Tryb pracy dla stopnia 2 lub 3
QUADRATIC_SPLINE = 0
CUBIC_SPLINE = 1

# Warunki brzegowe
FREE_BOUNDARY = 0
CLAMPED_BOUNDARY = 1
CUBIC_FUNCTION = 2


# Zadana funkcja f(x)
def f(x):
    global m, k
    return 10*m + (x**2)/k - 10*m*np.cos(k*x)


# Rysowanie pojedyńczej funkcji interpolującej
def make_spline(spline_mode, number_of_nodes, func, start_r, end_r, mode):

    X_base = np.linspace(start_r, end_r, 200)
    Y_base = func(X_base)

    # Testowe punkty dla interpolacji
    X_f = np.linspace(start_r, end_r, number_of_nodes)
    Y_f = func(X_f)

    if spline_mode == 0:
        solver = QuadraticSpline(X_f, Y_f, mode)
    else:
        solver = CubicSpline(X_f, Y_f, mode)

    X_res = np.linspace(start_r, end_r, 200)
    vec_f = np.vectorize(solver.compute)
    Y_res = vec_f(X_res)

    plt.plot(X_base, Y_base)
    plt.scatter(X_f, Y_f, c='red')
    plt.plot(X_res, Y_res, c='green')
    plt.show()


# Do rysowania interpolowanej funkcji
def plot_interpolating_function(func, start_r, end_r):
    X_base = np.linspace(start_r, end_r, 200)
    Y_base = func(X_base)
    plt.plot(X_base, Y_base)
    plt.title("Interpolowana funkcja")
    plt.grid(visible=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# Rysowanie błędów interpolacji dla zadanego stopnia i warunku brzegowego
def plot_error_of_an_interpolation(spline_mode, number_of_nodes, func, start_r, end_r, mode):
    X_base = np.linspace(start_r, end_r, 200)
    Y_base = func(X_base)

    # Testowe punkty dla interpolacji
    X_f = np.linspace(start_r, end_r, number_of_nodes)
    Y_f = func(X_f)

    if spline_mode == 0:
        solver = QuadraticSpline(X_f, Y_f, mode)
    else:
        solver = CubicSpline(X_f, Y_f, mode)

    X_res = np.linspace(start_r, end_r, 200)
    vec_f = np.vectorize(solver.compute)
    Y_res = vec_f(X_res)

    Y_error = np.abs(Y_res - Y_base)
    plt.plot(X_base, Y_error)
    plt.show()


# Pomocnicza funkcja do obliczania błędów
def compute_error_of_interpolation(Y_base, Y_res):
    sup_abs_diff = np.max(np.abs(Y_base - Y_res))
    sqr_sum_diff = np.sum(np.power(Y_base - Y_res, 2))

    print(f"Największy błąd bezwzględny: {sup_abs_diff}")
    print(f"Suma kwadratów różnic: {sqr_sum_diff}")


# Funkcja rysująca błędy dla zadanej liczby punktów dla wszystkich 2/3 warunków brzegowych
def error_comparison_for_all_modes(spline_mode, number_of_nodes, func, start_r, end_r):
    global error_points
    X_base = np.linspace(start_r, end_r, error_points)

    Y_base = func(X_base)

    X_f = np.linspace(start_r, end_r, number_of_nodes)
    Y_f = func(X_f)

    solvers = []
    vectorize_list = []
    X_res = np.linspace(start_r, end_r, error_points)
    Y_res = []
    Y_errors = []
    labels = ["free boundary", "clamped boundary", "cubic function"]

    if spline_mode == 0:
        spline_type = "QuadraticSpline"
        for i in range(2):
            solvers.append(QuadraticSpline(X_f, Y_f, i))
            vectorize_list.append(np.vectorize(solvers[-1].compute))
            Y_res.append(vectorize_list[-1](X_res))
            Y_errors.append(np.abs(Y_res[-1] - Y_base))
            plt.scatter(X_base, Y_errors[-1], label=labels[i], s=4)

            if i == 0:
                print(f"{spline_type} - Free boundary error:")
            else:
                print(f"{spline_type} - Clamped boundary error:")
            compute_error_of_interpolation(Y_base, Y_res[-1])
            print("")
    else:
        spline_type = "CubicSpline"
        for i in range(3):
            solvers.append(CubicSpline(X_f, Y_f, i))
            vectorize_list.append(np.vectorize(solvers[-1].compute))
            Y_res.append(vectorize_list[-1](X_res))
            Y_errors.append(np.abs(Y_res[-1] - Y_base))
            plt.scatter(X_base, Y_errors[-1], label=labels[i], s=4)

            if i == 0:
                print(f"{spline_type} - Free boundary error:")
            elif i == 1:
                print(f"{spline_type} - Clamped boundary error:")
            else:
                print(f"{spline_type} - Cubic function error:")
            compute_error_of_interpolation(Y_base, Y_res[-1])
            print("")

    plt.grid(visible=True)
    plt.title(f"{spline_type} - Error comparison for {number_of_nodes} points")
    plt.legend(loc="upper left")
    plt.show()


# Rysowanie dla zadanego stopnia wszystkich 2/3 funkcji interpolujących oraz oryginalnej funkcji
def spline_comparison(spline_mode, number_of_nodes, func, start_r, end_r):
    global error_points
    X_base = np.linspace(start_r, end_r, error_points)

    Y_base = func(X_base)

    X_f = np.linspace(start_r, end_r, number_of_nodes)
    Y_f = func(X_f)

    solvers = []
    vectorize_list = []
    X_res = np.linspace(start_r, end_r, error_points)
    Y_res = []
    labels = ["free boundary", "clamped boundary", "cubic function"]

    plt.plot(X_base, Y_base, label="f(x)")

    if spline_mode == 0:
        spline_type = "QuadraticSpline"
        for i in range(2):
            solvers.append(QuadraticSpline(X_f, Y_f, i))
            vectorize_list.append(np.vectorize(solvers[-1].compute))
            Y_res.append(vectorize_list[-1](X_res))
            plt.plot(X_base, Y_res[-1], label=labels[i])

    else:
        spline_type = "CubicSpline"
        for i in range(3):
            solvers.append(CubicSpline(X_f, Y_f, i))
            vectorize_list.append(np.vectorize(solvers[-1].compute))
            Y_res.append(vectorize_list[-1](X_res))
            plt.plot(X_base, Y_res[-1], label=labels[i])

    plt.scatter(X_f, Y_f, color='red')
    plt.grid(visible=True)
    plt.title(f"{spline_type} - Comparison for {number_of_nodes} points")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == '__main__':

    """
    *   Uruchomienie funkcji:
    *       error_comparison_for_all_modes - wykres różnic pomiędzy każdą funkcją interpolującą oraz funkcją
    *                                        interpolowaną
    *       argumenty:  [stopień funkcji sklejanej]: int - zdefiniowane na górze kodu makra
    *                   [ilość punktów]: int
    *                   [funkcja f(x)]: func - funkcja, która jest określona na badanym przedziale
    *                   [początek badanego obszaru]: int
    *                   [koniec badanego obszaru]: int
    *
    *       spline_comparison - wykres wszystkich funkcji interpolujących dla zadanego stopnia oraz funkcja interpolowana
    *       argumenty:  dokładnie takie same, jak dla error_comparison_for_all_modes
    *
    *       :ostrzeżenie: CUBIC_FUNCTION oczywiście nie zadziała dla trybu QUADRATIC_SPLINE       
    """
    plot_interpolating_function(f, range_start, range_end)
    exit(0)

    error_comparison_for_all_modes(QUADRATIC_SPLINE, 5, f, range_start, range_end)
    spline_comparison(QUADRATIC_SPLINE, 5, f, range_start, range_end)

    error_comparison_for_all_modes(CUBIC_SPLINE, 30, f, range_start, range_end)
    spline_comparison(CUBIC_SPLINE, 30, f, range_start, range_end)
