
import numpy as np


# Funkcje sklejane 2 stopnia (kwadratowe)
class QuadraticSpline:
    def __init__(self, x_s, y_s, boundary_condition):
        self.n = len(y_s)
        self.x_s = x_s
        self.y_s = y_s
        # funkcji będzie n-1 przy n punktach
        self._a = None
        self._b = None    # współczynników b będzie n, żeby obliczyć c(n-1)
        self._c = None
        self._boundary_condition = boundary_condition   # natural = 0
        self._solved_a = False
        self._solved_b = False
        self._solved_c = False
        self.res_functions = [None for _ in range(self.n-1)]
        self._solve()

    # (16)
    def gamma(self, j):
        return (self.y_s[j] - self.y_s[j-1]) / (self.x_s[j] - self.x_s[j-1])

    # (14)
    def c(self, j):
        if len(self._b) == 0:
            print("Współczynniki b nie zostały jeszcze obliczone!")
            return None

        return (self._b[j+1] - self._b[j]) / (2*(self.x_s[j+1] - self.x_s[j]))

    def b(self, j):
        return 2*self.gamma(j) - self._b[j-1]

    def _solve_for_b_coefficient(self):
        if self._boundary_condition == 0:
            self._b = [0]

            for j in range(1, self.n):
                self._b.append(2*self.gamma(j) - self._b[-1])

            self._solved_b = True

        elif self._boundary_condition == 1:
            self._b = [self.gamma(1)]

            for j in range(1, self.n):
                self._b.append(2*self.gamma(j) - self._b[-1])

            self._solved_b = True

    def _solve_a(self):
        self._a = self.y_s
        self._solved_a = True

    def _solve_c(self):
        self._c = [self.c(j) for j in range(self.n-1)]

    def make_functions(self):
        for i in range(self.n-1):
            # ci (x - xi)^2
            c_pol = self._c[i] * np.polymul([-self.x_s[i], 1], [-self.x_s[i], 1])

            # bi (x - xi)
            b_pol = [self._b[i] * -self.x_s[i], self._b[i] * 1, 0]

            # ai
            a_pol = [self._a[i], 0, 0]

            self.res_functions[i] = np.polynomial.polynomial.Polynomial(np.polyadd(np.polyadd(a_pol, b_pol), c_pol))

    def _solve(self):
        self._solve_for_b_coefficient()
        self._solve_a()
        self._solve_c()
        self.make_functions()

    def compute(self, x_val):
        for i in range(len(self.res_functions)):
            if self.x_s[i] <= x_val <= self.x_s[i + 1]:
                return self.res_functions[i](x_val)
