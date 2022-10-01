
import numpy as np


# Funkcje sklejane 3 stopnia
class CubicSpline:
    def __init__(self, x_s, y_s, boundary_condition):
        self.n = len(y_s)
        self.x_s = x_s
        self.y_s = y_s
        # funkcji będzie n-1 przy n punktach
        self._a = None
        self._b = None
        self._c = None
        self._d = None
        self._sigma = None  # σ
        self._boundary_condition = boundary_condition
        self._solved_a = False
        self._solved_b = False
        self._solved_c = False
        self._solved_d = False
        self.res_functions = [None for _ in range(self.n-1)]
        self._solve()

    def h(self, i):
        if i == self.n-1:
            print("Błąd - próba obliczenia h[n-1], do której potrzeba x_s[n]")
        return self.x_s[i+1] - self.x_s[i]

    # Normalne ∆ (bez górnego indeksu)
    def delta(self, i):
        if i == self.n-1:
            print("Błąd - próba obliczenia delta[n-1], do której potrzeba y_s[n]")
        return (self.y_s[i+1] - self.y_s[i]) / self.h(i)

    def make_progressive_difference_table(self, x_s_diff, y_s_diff):
        n_diff = len(y_s_diff)
        prog_diff = list(y_s_diff)

        for y in range(1, n_diff):
            for x in range(n_diff-1, y-1, -1):
                prog_diff[x] = (prog_diff[x] - prog_diff[x-1]) / (x_s_diff[x] - x_s_diff[x - y])

        return prog_diff

    def _solve_sigma(self):

        # indeksy w porównaniu do wykładu przy h są o 1 mniejsze (tam jest indeksowanie od 1)
        if self._boundary_condition == 0:       # free boundary
            A_tab = [[0 for _ in range(self.n-2)] for _ in range(self.n-2)]

            for i in range(self.n-2):
                A_tab[i][i] = 2 * (self.h(i) + self.h(i+1))
                if i-1 >= 0:
                    A_tab[i][i - 1] = self.h(i)
                if i+1 != self.n-2:
                    A_tab[i][i + 1] = self.h(i+1)

        elif self._boundary_condition == 1:     # clamped boundary
            A_tab = [[0 for _ in range(self.n)] for _ in range(self.n)]

            for i in range(1, self.n - 1):
                A_tab[i][i] = 2 * (self.h(i - 1) + self.h(i))
                A_tab[i][i - 1] = self.h(i - 1)
                A_tab[i][i + 1] = self.h(i)

            A_tab[0][0] = 2
            A_tab[0][1] = 1
            A_tab[-1][-2] = 2
            A_tab[-1][-1] = 1
        elif self._boundary_condition == 2:     # cubic function
            A_tab = [[0 for _ in range(self.n)] for _ in range(self.n)]

            for i in range(1, self.n - 1):
                A_tab[i][i] = 2 * (self.h(i - 1) + self.h(i))
                A_tab[i][i - 1] = self.h(i - 1)
                A_tab[i][i + 1] = self.h(i)

            A_tab[0][0] = -self.h(0)
            A_tab[0][1] = self.h(0)
            A_tab[-1][-2] = self.h(self.n-2)
            A_tab[-1][-1] = -self.h(self.n-2)
        else:
            print(f"Niepoprawny boundary_condition: {self._boundary_condition}")
            return

        B_tab = [0 for _ in range(self.n)]
        for i in range(1, self.n-1):
            B_tab[i] = self.delta(i) - self.delta(i-1)

        if self._boundary_condition == 0:       # free boundary
            B_tab = B_tab[1:-1]
        elif self._boundary_condition == 1:     # clamped boundary
            # przybliżenie f'1 za pomocą ilorazu różnicowego (y1 - y0) / (x1 - x0) i analogicznie dla f'n-1
            B_tab[0] = (self.delta(1) - ((self.y_s[1] - self.y_s[0]) / (self.x_s[1] - self.x_s[0]))) / self.h(1)
            B_tab[-1] = (self.delta(self.n-2) - ((self.y_s[-2] - self.y_s[-1]) / (self.x_s[-2] - self.x_s[-1]))) / self.h(self.n-2)
        elif self._boundary_condition == 2:     # cubic function
            # W przypadku funkcji sześciennej bierzemy tylko 4 punkty z początku i 4 punkty z końca
            prog_diff_start = self.make_progressive_difference_table(self.x_s[:4], self.y_s[:4])
            prog_diff_end = self.make_progressive_difference_table(self.x_s[-4:], self.y_s[-4:])
            B_tab[0] = self.h(0)**2 * prog_diff_start[3]
            B_tab[-1] = -self.h(self.n-2)**2 * prog_diff_end[3]

        A = np.array(A_tab)
        B = np.array(B_tab)

        res_sigma = np.linalg.solve(A, B)
        if self._boundary_condition == 0:
            res_sigma = np.insert(res_sigma, 0, 0., axis=0)
            res_sigma = np.append(res_sigma, 0)

        self._sigma = res_sigma

    def _solve_a(self):
        self._a = self.y_s
        self._solved_a = True

    def _solve_b(self):
        self._b = [0 for _ in range(self.n-1)]
        for i in range(self.n-1):
            self._b[i] = (self.y_s[i+1] - self.y_s[i]) / self.h(i) - self.h(i)*(self._sigma[i+1] + 2*self._sigma[i])
        self._solved_b = True

    def _solve_c(self):
        self._c = [0 for _ in range(self.n-1)]
        for i in range(self.n-1):
            self._c[i] = 3 * self._sigma[i]
        self._solved_c = True

    def _solve_d(self):
        self._d = [0 for _ in range(self.n-1)]
        for i in range(self.n-1):
            self._d[i] = (self._sigma[i+1] - self._sigma[i]) / self.h(i)

    def make_functions(self):
        for i in range(self.n-1):

            # di (x - xi)^3
            d_pol = self._d[i] * np.polymul(np.polymul([-self.x_s[i], 1], [-self.x_s[i], 1]), [-self.x_s[i], 1])   # 4 współczynniki

            # ci (x - xi)^2
            c_pol = self._c[i] * np.polymul([-self.x_s[i], 1], [-self.x_s[i], 1, 0])    # 3 współ. + 0

            # bi (x - xi)
            b_pol = [self._b[i] * -self.x_s[i], self._b[i] * 1, 0, 0]   # 2 współ. + 2 * 0
            # ai
            a_pol = [self._a[i], 0, 0, 0]   # 1 współ. + 3 * 0

            self.res_functions[i] = np.polynomial.polynomial.Polynomial(np.polyadd(np.polyadd(np.polyadd(a_pol, b_pol), c_pol), d_pol))

    def _solve(self):
        self._solve_sigma()
        self._solve_a()
        self._solve_b()
        self._solve_c()
        self._solve_d()
        self.make_functions()

    def compute(self, x_val):
        for i in range(len(self.res_functions)):
            if self.x_s[i] <= x_val <= self.x_s[i + 1]:
                return self.res_functions[i](x_val)
