
import numpy as np


class TrigonometricApproximation:
    def __init__(self, x_s, y_s, w_s, m):
        self.x_s = x_s
        self.y_s = y_s
        self.w_s = w_s
        self.n = len(x_s)
        self.range_length = self.x_s[-1] - self.x_s[0]
        self.range_start = self.x_s[0]

        if m > ((self.n - 1) // 2):
            raise Exception("TrigonometricApproximation - za duze m!")
        self.m = m

        self.a = [0 for _ in range(self.m)]
        self.b = [0 for _ in range(self.m)]
        self.approx_func = None
        self.solved = False

    def solve_a_coefficient(self):
        for j in range(self.m):
            self.a[j] = (2 / self.n) * np.sum([self.y_s[i] * np.cos(j * self.x_s[i]) for i in range(self.n)])

    def solve_b_coefficient(self):
        for j in range(self.m):
            self.b[j] = (2 / self.n) * np.sum([self.y_s[i] * np.sin(j * self.x_s[i]) for i in range(self.n)])

    def solve(self):
        self.scale_points_to_pi_range()
        self.solve_a_coefficient()
        self.solve_b_coefficient()
        self.approx_func = lambda x: (self.a[0] / 2) + np.sum([self.a[j]*np.cos(j * x) + self.b[j]*np.sin(j * x) for j in range(1, self.m)])
        self.scale_points_from_pi_range()
        self.solved = True

    def scale_points_to_pi_range(self):
        for i in range(self.n):
            self.x_s[i] -= self.range_start     # przeskalowanie o początek przedziału
            self.x_s[i] /= self.range_length
            self.x_s[i] *= 2 * np.pi
            self.x_s[i] -= np.pi

    def scale_points_from_pi_range(self):
        for i in range(self.n):
            self.x_s[i] += np.pi
            self.x_s[i] /= 2 * np.pi
            self.x_s[i] *= self.range_length
            self.x_s[i] += self.range_start

    def scale_range_to_pi(self, x):
        x -= self.range_start
        x /= self.range_length
        x *= 2 * np. pi
        x -= np.pi
        return x

    def calc(self, x):
        x_scaled = self.scale_range_to_pi(np.copy(x))
        vf = np.vectorize(self.approx_func)

        return vf(x_scaled)








