
import numpy as np


class PolynomialApproximation:
    def __init__(self, x_s, y_s, w_s, m):
        self.x_s = x_s
        self.y_s = y_s
        self.w_s = w_s
        self.m = m
        self.n = len(x_s)
        self.solved = False
        self.A = None
        self.approx_func = None

    def solve(self):
        G_elements = [0 for _ in range(2*self.m+1)]
        # i - indeks punktu (od 0 do n)
        # k - potęga przy xi (od 0 do m)
        for k in range(2*self.m+1):
            G_elements[k] = np.sum([self.w_s[i] * self.x_s[i] ** k for i in range(self.n)])

        G_tab = [[0 for _ in range(self.m)] for _ in range(self.m)]

        for y in range(self.m):
            for x in range(self.m):
                G_tab[y][x] = G_elements[x+y]

        B_tab = [0 for _ in range(self.m)]

        for k in range(self.m):
            B_tab[k] = np.sum([self.w_s[i] * self.y_s[i] * self.x_s[i] ** k for i in range(self.n)])

        G = np.array(G_tab)
        B = np.array(B_tab).reshape(-1, 1)
        self.A = np.linalg.solve(G, B)

        poly = []
        for a in self.A:
            poly.append(*a)

        self.approx_func = np.polynomial.Polynomial(poly)
        self.solved = True

    def calc(self, x):
        return self.approx_func(x)

