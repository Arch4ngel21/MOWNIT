import numpy as np
import matplotlib.pyplot as plt
import random

m_s = []
x_s = []
f_x = []    # i_ta tablica w f_x odpowiada m_i wartosciom pochodnych w punkcie x_i

# Zdefiniowane stałe dla zadanej funkcji
m = 1
k = 2
range_start = np.double(-3) * np.pi
range_stop = np.double(3) * np.pi
max_derivative = 2

# Liczba punktów przy obliczaniu błędów interpolacji
error_points = 500


# Zadana funkcja
def f(x):
    return 10 * m + x**2 / k - 10 * m * np.cos(k * x)


def f_1(x):
    return 2 * x / k + 10 * m * k * np.sin(k * x)


def f_2(x):
    return 2 / k + 10 * m * k**2 * np.cos(k * x)


def f_3(x):
    return -10 * m * k**3 * np.sin(k * x)


def f_4(x):
    return -10 * m * k**4 * np.cos(k * x)


# Funkcja generująca punkty Czebyszewa
def chebyshev_points(a, b, n: int):

    x = np.arange(n, dtype=np.double)
    return np.double(0.5) * (a + b) + np.double(0.5) * (b - a) * np.cos((2*x + 1) * np.pi / (2*n))


def factorial(n):
    if n <= 1:
        return 1

    temp = [1]
    for i in range(2, n):
        temp.append(temp[-1] * i)
    return temp[-1]


def make_table_for_b_coefficients():
    n = sum(m_s)
    diff_table = [[0 for _ in range(n+1)] for _ in range(n)]    # dodana 1 kolumna z lewej, zeby pamietac jaki x jest w danym rzedzie
    table_pos = 0
    for e, m_val in enumerate(m_s):
        for _ in range(m_val):
            diff_table[table_pos][0] = e    # w pierwszej kolumnie zapisane sa indeksy xksow, ktre znajduja sie w odpowiednim rzedzie
            table_pos += 1

    same_x = 1
    for x in range(1, n+1):
        for y in range(x-1, n):
            if y != 0 and diff_table[y][0] == diff_table[y-1][0]:
                same_x += 1
            else:
                same_x = 1

            if m_s[diff_table[y][0]] > x-1 and same_x >= x-1:
                diff_table[y][x] = f_x[diff_table[y][0]][x-1] / factorial(x-1)
                # same_x += 1

            else:
                x0_index = (y - (x-1))
                xk_index = x0_index + (x-1)
                diff_table[y][x] = (diff_table[y][x-1] - diff_table[y-1][x-1]) / (x_s[diff_table[xk_index][0]] - x_s[diff_table[x0_index][0]])
                if np.isnan(diff_table[y][x]):
                    print(f"Nan w {x, y} : {x_s[diff_table[xk_index][0]]} - {x_s[diff_table[x0_index][0]]}, indeksy: {xk_index, x0_index}")

    return diff_table


def make_factor(i):
    global x_s, m_s

    res_func = [1]
    x_index = 0
    while i > 0:
        if i - m_s[x_index] >= 0:
            factor_power = m_s[x_index]
        else:
            factor_power = i
        new_factor = np.polynomial.polynomial.polypow(np.poly1d([1, -x_s[x_index]]), factor_power)
        if x_s[x_index] != 0:
            res_func = np.polymul(res_func, new_factor)
        else:
            res_func = np.pad(res_func, (0, 1))
        print(f"(x-{x_s[x_index]})^{factor_power}", end=" ")
        i -= m_s[x_index]
        x_index += 1

    print("")
    # print("New factor:", res_func)
    return res_func


class HermitInterpolation:
    def __init__(self):
        self.diff_table = None
        self.m = 0
        self.n = 0

    def hermit_interpolation(self, m_s_input, x_s_input, f_x_input):
        global m_s
        global x_s
        global f_x

        m_s = m_s_input
        x_s = x_s_input
        f_x = f_x_input

        self.m = sum(m_s)
        self.n = self.m - 1

        xs_ = []
        for i in range(len(x_s)):
            xs_.extend([x_s[i]] * m_s[i])

        self.diff_table = [[None] * self.m for _ in range(self.m)]

        i = 0
        for y_list in f_x:
            for j in range(len(y_list)):
                for k in range(j + 1):
                    self.diff_table[i][k] = y_list[k] / factorial(k)
                i += 1

        # Fill the remaining triangular part of a matrix
        for j in range(1, self.m):
            for i in range(j, self.m):
                if self.diff_table[i][j] is not None:
                    continue
                self.diff_table[i][j] = (self.diff_table[i][j - 1] - self.diff_table[i - 1][j - 1]) / (xs_[i] - xs_[i - j])

    def calc(self, x):
        global x_s, f_x, m_s
        y_val = self.diff_table[0][0]
        polynomial = 1
        degree = 0
        for i, mi in enumerate(m_s):
            for _ in range(mi):
                degree += 1
                polynomial *= x - x_s[i]
                y_val += self.diff_table[degree][degree] * polynomial
                if degree == self.n:
                    return y_val


# Wyznaczenie błędów interpolacji
def check_accuracy(start: np.double, end: np.double, f, vec_f: np.vectorize):
    x = np.linspace(start, end, error_points)
    f_y = f(x)
    interpolated_y = vec_f(x)

    return np.ndarray.max(np.absolute(np.subtract(f_y, interpolated_y))), np.sum(np.power(np.subtract(f_y, interpolated_y), np.double(2.0)))


def test(n, type):
    global range_start, range_stop, max_derivative
    func = [f, f_1, f_2, f_3, f_4]

    if type == "regular":
        X_base = np.linspace(range_start, range_stop, n)
        points_label = "Regular Points"
    else:
        X_base = chebyshev_points(range_start, range_stop, n)
        points_label = "Chebyshev Points"

    Y_base = [[] for _ in range(n)]
    M_base = [0 for _ in range(n)]

    for i in range(n):
        # derivative = random.randint(0, max_derivative)
        derivative = max_derivative
        M_base[i] = derivative
        if not derivative:
            M_base[i] = 1

        Y_base[i].append(func[0](X_base[i]))
        for j in range(1, derivative):
            Y_base[i].append(func[j](X_base[i]))

    solver = HermitInterpolation()
    solver.hermit_interpolation(M_base, X_base, Y_base)
    vec_f = np.vectorize(solver.calc)

    X_plot = np.linspace(range_start, range_stop, 500)
    Y_plot = f(X_plot)
    Y_res = vec_f(X_plot)
    Y_base_plot = np.array([y[0] for y in Y_base])

    err_res = check_accuracy(range_start, range_stop, f, vec_f)

    print(f"Bledy interpolacji dla n = {n} ({type}) : max_diff = {err_res[0]}, sqr_diff = {err_res[1]}")

    plt.plot(X_plot, Y_plot, c='blue')
    plt.plot(X_plot, Y_res, c='green')
    plt.scatter(X_base, Y_base_plot, c='red')
    plt.title("Hermit Interpolation | " + points_label + f" | n = {n}")
    plt.show()


if __name__ == '__main__':
    n_s = [3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]

    test(5, "regular")
    test(10, "chebyshev")

    for n in n_s:
        test(n, "regular")

