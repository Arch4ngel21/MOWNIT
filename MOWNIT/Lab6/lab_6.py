# Rozwiązywanie układów równań i układów równań nieliniowych
from typing import List, Optional
import numpy as np
from numpy.linalg import inv
import sympy
import matplotlib.pyplot as plt


n = 12
m = 20
r_start = 0.2
r_end = 1.8
sigma = 1e-04


def f(x):
    global n, m
    return x**2 - m*np.power(np.sin(x), n)


def f_(x):
    global n, m
    return 2*x - m * n * np.power(np.sin(x), n - 1) * np.cos(x)


def stop_func_value(curr_value, last_value):
    global sigma
    return abs(f(curr_value)) < sigma


def stop_func_step(curr_value, last_value):
    global sigma
    return abs(last_value - curr_value) < sigma


class NonLinearEquation:
    def __init__(self, stop_func, range_start, range_stop, mode, starting_point=0):

        if mode == 0:
            self.current_res = starting_point
        else:
            self.current_res = starting_point
        self.stop_criteria = stop_func
        self.range_start = range_start
        self.range_stop = range_stop
        self.stop_func = stop_func
        self.last_res = range_stop
        self.iterations = 0

    def newton_method(self):
        while not self.stop_func(self.current_res, self.last_res):
            if np.isnan(self.current_res) or self.iterations == 5000:
                break

            self.iterations += 1
            print(self.current_res)
            self.current_res -= f(self.current_res) / f_(self.current_res)
        print(f"Estimated result: {self.current_res}")

    def secant_method(self):
        temp = self.current_res
        self.current_res -= (f(self.current_res) * (self.current_res - self.last_res)) / (f(self.current_res) - f(
            self.last_res))
        self.last_res = temp

        while not self.stop_func(self.current_res, self.last_res):
            # print(self.current_res)
            if np.isnan(self.current_res) or self.iterations == 5000:
                break
            self.iterations += 1
            temp = self.current_res
            self.current_res -= (f(self.current_res) * (self.current_res - self.last_res)) / f(self.last_res) - f(self.current_res)
            self.last_res = temp
        print(f"Estimated result: {self.current_res}")


class NonLinearSystem:
    def __init__(self, starting_vector, F, symb, sigma):
        self.starting_vector = starting_vector
        self.X = starting_vector
        self.k = 3
        self.F = F
        self.symb = symb
        self.sigma = sigma
        self.J: Optional[List] = None
        self.iterations = 0

    def make_jacobian(self):
        self.J = [[0 for _ in range(self.k)] for _ in range(len(self.F))]

        for f in range(self.F.shape[0]):
            self.J[f][0] = sympy.diff(*self.F[f], self.symb[0])
            self.J[f][1] = sympy.diff(*self.F[f], self.symb[1])
            self.J[f][2] = sympy.diff(*self.F[f], self.symb[2])

    def compute_jacobian(self):
        new_J = [[0 for _ in range(self.k)] for _ in range(len(self.F))]

        for var in range(self.k):
            for f in range(self.F.shape[0]):
                new_J[f][var] = self.J[f][var].subs(self.symb[0], self.X[0][0])
                new_J[f][var] = new_J[f][var].subs(self.symb[1], self.X[1][0])
                new_J[f][var] = new_J[f][var].subs(self.symb[2], self.X[2][0])

        return new_J

    def newton_method(self):
        # J = [[0 for _ in range(self.k)] for _ in range(len(self.F))]
        self.make_jacobian()

        while self.iterations < 50:
            # print("\nJacobian:")
            # for lane in self.J:
            #     print(lane)

            # print("\nSubbed:")
            subbed_J = self.compute_jacobian()
            # for lane in subbed_J:
            #     print(lane)

            F_x = [[0] for _ in range(self.F.shape[0])]
            check_sum = []
            for f in range(self.F.shape[0]):
                F_x[f][0] = self.F[f][0].subs(self.symb[0], self.X[0][0])
                F_x[f][0] = F_x[f][0].subs(self.symb[1], self.X[1][0])
                F_x[f][0] = F_x[f][0].subs(self.symb[2], self.X[2][0])
                check_sum.append(abs(F_x[f][0]))

            subbed_J = np.array(subbed_J, dtype=np.float64)

            # print("Before conversion:")
            # print(F_x)
            F_x = np.array(F_x, dtype=np.float64)

            # print("\bBefore:")
            # print(F_x)
            # print(inv(subbed_J))

            # print("\nAfter:")
            # print(self.X)
            # print(inv(subbed_J) @ F_x)

            self.X -= inv(subbed_J) @ F_x

            # print(f"{i}:", check_sum, " X:", self.X)
            check_sum = np.array(check_sum)
            if all(check_sum < self.sigma):
                # print("Check_sum =", check_sum)
                # print(f"Ended on {i} step")
                return

            self.iterations += 1


def tests():
    step = 0.1
    starting_point = r_start + step
    # res_file = open("res_lab_secant_4.txt", "w")

    while starting_point < r_end:
        print("Starting point = ", starting_point)
        solver2 = NonLinearEquation(stop_func_value, r_start, r_end, 0, starting_point)
        solver2.newton_method()
        # res_file.write(str(starting_point) + " " + str(solver2.iterations) + " " + str(solver2.current_res) + "\n")
        starting_point += step

    # res_file.write("\n")

    # res_file.close()


# Do rysowania interpolowanej funkcji
def plot_function():
    X_base = np.linspace(r_start, r_end, 200)
    Y_base = f(X_base)
    plt.plot(X_base, Y_base, c="green", label="Funkcja")
    plt.plot(X_base, np.zeros(200), c="red", label="Oś OX")
    plt.title("Wykres zadanej funkcji")
    plt.grid(visible=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="lower left")
    plt.show()


# Funkcja używana do generowania wykresów 3d
def plot_3d_result(green_points, red_points, green_solutions):
    ax = plt.axes(projection="3d")

    # x_line_g = np.array([x[0] for x in green_points])
    # y_line_g = np.array([x[1] for x in green_points])
    # z_line_g = np.array([x[2] for x in green_points])

    # x_line_r = np.array([x[0] for x in red_points])
    # y_line_r = np.array([x[1] for x in red_points])
    # z_line_r = np.array([x[2] for x in red_points])

    # ax.scatter3D(x_line_g, y_line_g, z_line_g, cmap="Greens", label="Found solution")
    # ax.scatter3D(x_line_r, y_line_r, z_line_r, cmap="Reds", label="Couldn't find solution")
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.title("Result for generated points")
    # plt.legend(loc="lower left")

    # plt.show()

    cat = [[] for _ in range(4)]
    colors = ["Oranges", "Purples", "Greens", "YlGn"]
    p1 = [-1, 1, 1]
    p2 = [0.5, 1.0, 0.5]
    p3 = [0.5, 1.0, -0.5]
    p4 = [-1, 1, -1]

    for i in range(len(green_solutions)):
        if same_point(green_solutions[i], p1):
            cat[0].append(green_points[i])
        elif same_point(green_solutions[i], p2):
            cat[1].append(green_points[i])
        elif same_point(green_solutions[i], p3):
            cat[2].append(green_points[i])
        else:
            cat[3].append(green_points[i])

    # x_line, y_line, z_line = [], [], []
    for i in range(4):
        x_line, y_line, z_line = [], [], []
        for j in range(len(cat[i])):
            x_line.append(cat[i][j][0])
            y_line.append(cat[i][j][1])
            z_line.append(cat[i][j][2])
        ax.scatter3D(x_line, y_line, z_line, cmap=colors[i], label=f"Category {i}")
    plt.legend(loc="lower left")
    plt.show()


def new_solution(solutions, new_x, sigma):
    for i in range(len(solutions)):
        flag = 1

        for x_i in range(len(new_x)):
            if abs(new_x[x_i] - solutions[i][x_i]) > sigma:
                flag = 0
                break

        if flag == 1:
            return False

    return True


def same_point(x1, x2):
    for j in range(len(x1)):
        if abs(x1[j] - x2[j]) > sigma:
            return False
    return True


if __name__ == '__main__':

    tests()
    exit(0)

    x, y, z = sympy.symbols('x y z')

    f1 = x**2 + y**2 - z**2 - 1
    f2 = x - 2*y**3 + 2*z**2 + 1
    f3 = 2*x**2 + y - 2*z**2 - 1

    F = np.array([f1, f2, f3]).reshape(-1, 1)

    green_points = []
    red_points = []
    solutions = []
    green_solutions = []
    average_number_of_iterations = 0

    for j in range(1000):
        try:
            # print("\n")
            z_axis = np.random.uniform(-1000, 1000)
            start = np.array([[np.random.uniform(-1000, 1000)], [np.random.uniform(-1000, 1000)], [z_axis - z_axis % 100]], dtype=np.float64)
            start_point = np.copy(start)

            solver = NonLinearSystem(start, F, [x, y, z], sigma)
            solver.newton_method()

            # print(solver.X)

            f1_val = F[0][0].subs(x, solver.X[0][0])
            f1_val = f1_val.subs(y, solver.X[1][0])
            f1_val = f1_val.subs(z, solver.X[2][0])
            f2_val = F[1][0].subs(x, solver.X[0][0])
            f2_val = f2_val.subs(y, solver.X[1][0])
            f2_val = f2_val.subs(z, solver.X[2][0])
            f3_val = F[2][0].subs(x, solver.X[0][0])
            f3_val = f3_val.subs(y, solver.X[1][0])
            f3_val = f3_val.subs(z, solver.X[2][0])
            # print(f"F1: {f1_val}, F2: {f2_val}, F3: {f3_val}")
            if all(np.array([f1_val, f2_val, f3_val]) < sigma):
                print(f"{j}, Got solution:")
                green_points.append(start_point)
                green_solutions.append(solver.X)
                average_number_of_iterations += solver.iterations

                if new_solution(solutions, solver.X, sigma):
                    solutions.append(solver.X.tolist())

            else:
                red_points.append(start_point)

        except np.linalg.LinAlgError as err:
            red_points.append(start_point)
            continue

    print("All found solutions:")
    print(solutions)
    average_number_of_iterations /= len(green_points)
    print("Average number of iterations for result:", average_number_of_iterations)

    plot_3d_result(green_points, red_points, green_solutions)
