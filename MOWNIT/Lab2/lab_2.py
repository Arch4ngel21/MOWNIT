import numpy as np
import matplotlib.pyplot as plt
import scipy.special


# Zdefiniowane stałe dla zadanej funkcji
m = 1
k = 2
range_start = np.double(-3) * np.pi
range_stop = np.double(3) * np.pi

# Liczba punktów przy obliczaniu błędów interpolacji
error_points = 500


# Zadana funkcja
def f(x):
    return 10 * m + x**2 / k - 10 * m * np.cos(k * x)


# Metoda Lagrange'a
def lagrange_polynomial(x, points):

    return np.sum([p1[1] * np.prod(np.array([(x - p2[0]) / (p1[0] - p2[0]) for p2 in points if p1[0] != p2[0]])) for p1 in points])


# Funkcja generująca punkty Czebyszewa
def chebyshev_points(a, b, n: int):

    x = np.arange(n, dtype=np.double)
    return np.double(0.5) * (a + b) + np.double(0.5) * (b - a) * np.cos((2*x + 1) * np.pi / (2*n))


# Pomocnicza funkcja dla metody Newtona - tworzy tablicę ilorazów różnicowych
def make_difference_quotient_table(pts):
    table = [[None for _ in range(len(pts))] for _ in range(len(pts))]

    for i in range(len(pts)):
        table[i][0] = pts[i][1]

    for i in range(1, len(pts)):
        for j in range(1, len(pts)):
            if i >= j:
                table[i][j] = (table[i][j-1] - table[i-1][j-1]) / (pts[i][0] - pts[i-j][0])

    return table


# Metoda Newtona
def newton_polynomial(x, diff_table, points):

    return diff_table[0][0] + np.sum([diff_table[i][i] * np.prod([(x - points[k][0]) for k in range(i)]) for i in range(1, len(points))])


def make_forward_diff_table(pts):

    table = [[None for _ in range(len(pts))] for _ in range(len(pts))]

    for i in range(len(pts)):
        table[i][0] = pts[i][1]

    for k in range(1, len(pts)):
        for i in range(len(pts)-k):
            table[i][k] = table[i+1][k-1] - table[i][k-1]

    return table


# Metoda Newtona dla węzłów równoodległych (jako dodatek, ale nie załączałem testów)
def newton_polynomial_spaced(x, x0, forward_diff_table, h, points):

    epsilion = 1e-08        # zadane epsilion ze względu na błędy obliczeniowe
    s = (x - x0) / h
    print(s)

    if not s.is_integer():
        if np.abs(s - np.round(s)) < epsilion:
            np.round(s)
        else:
            print("Podano błędny argument x!")
            return None

    return np.sum([scipy.special.binom(s, k) * forward_diff_table[0][k] for k in range(len(points))])


# Wyznaczenie błędów interpolacji
def check_accuracy(start: np.double, end: np.double, f, vec_f: np.vectorize, mode: chr, points=None, diff_table=None):
    x = np.linspace(start, end, error_points)
    f_y = f(x)
    if mode == 'l':
        interpolated_y = vec_f(x, points=points)
    elif mode == 'n':
        interpolated_y = vec_f(x, diff_table=diff_table, points=points)
    else:
        print("Wrong mode!")
        return

    return np.ndarray.max(np.absolute(np.subtract(f_y, interpolated_y))), np.sum(np.power(np.subtract(f_y, interpolated_y), np.double(2.0)))


# Funkcja dla interpolacji obiema metodami
def interpolate(start: np.double, end: np.double, n: int, f, points_mode: chr, interpolate_mode: str):
    if points_mode == 'e':
        points_label = " | Evenly spaced points"
        x = np.linspace(start, end, n)
    elif points_mode == 'c':
        points_label = " | Chebyshev points"
        x = chebyshev_points(start, end, n)
    else:
        print("Wrong points_mode argument!")
        return

    y = f(x)
    points = list(zip(x, y))
    f_x = np.linspace(start, end, 200)
    if interpolate_mode == "lagrange":
        label = "Lagrange polynomial"
        vec_f = np.vectorize(lagrange_polynomial, excluded=['points'])
        f_y = vec_f(f_x, points=points)
        acc, acc_sqr = check_accuracy(start, end, f, vec_f, interpolate_mode[0], points)
        print("\nMax value error: ", acc, "\nSquare Error: ", acc_sqr)

    elif interpolate_mode == "newton":
        label = "Newton Polynomial"
        table = make_difference_quotient_table(points)
        vec_f = np.vectorize(newton_polynomial, excluded=['diff_table', 'points'])
        f_y = vec_f(f_x, diff_table=table, points=points)
        acc, acc_sqr = check_accuracy(start, end, f, vec_f, interpolate_mode[0], points, table)
        print("\nMax value error: ", acc, "\nSquare Error: ", acc_sqr)
    else:
        print("Wrong interpolate_mode argument!")
        return

    plot_f_x = np.linspace(start, end, 100)
    plot_f_y = f(plot_f_x)

    plt.plot(plot_f_x, plot_f_y)
    plt.scatter(x, y, c='red')
    plt.plot(f_x, f_y, c='green')
    plt.title(label + points_label + f" | n = {n}")
    plt.show()
    return acc_sqr


# Funkcja dla zapisywania błędów interpolacji z testów
def tests(start, end, f):
    n_s = [3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100]
    file = open("res_lab2_sqr.txt", 'w')
    for n in n_s:
        file.write(str(interpolate(start, end, n, f, 'c', 'lagrange')))
        file.write(" ")
        file.write(str(interpolate(start, end, n, f, 'c', 'newton')))
        file.write(" ")
        file.write(str(interpolate(start, end, n, f, 'e', 'lagrange')))
        file.write(" ")
        file.write(str(interpolate(start, end, n, f, 'e', 'newton')))
        file.write("\n")

    file.close()


if __name__ == '__main__':

    interpolate(range_start, range_stop, 20, f, 'c', "lagrange")
    interpolate(range_start, range_stop, 15, f, 'c', "newton")
    interpolate(range_start, range_stop, 10, f, 'e', "lagrange")
    interpolate(range_start, range_stop, 40, f, 'e', "newton")

    # W celu zobaczenia testów dla wszystkich n, które zostały przedstawione w prezentacji, odkomentować
    # ns = [3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100]

    # for n_val in ns:
    #     interpolate(range_start, range_stop, n_val, f, 'c', "newton")

