import numpy as np

"""
*   numpy.single - 8 bits exponent, 23 bits mantissa
*   numpy.double - 11 bits exponent, 52 bits mantissa
"""


def f(x):
    return np.sin(x) + np.cos(3*x)


def f_(x, h):
    return (f(x + h) - f(x)) / h


def f_exact(x):
    return np.cos(x) - 3*np.sin(3*x)


if __name__ == '__main__':

    # file = open("res_lab_1.txt", "w")

    print("\nSingle precision f'(x) value:\n")
    x0 = np.single(1.0)
    base = np.single(0.5)

    for n in range(41):
        h = np.power(base, n, dtype=np.single)
        print(f"h: {h} -> f'(x): ", f_(x0, h))
        # file.write(str(h) + " " + str(f_(x0, h)) + "\n")

    print("\nDouble precision f'(x) value:\n")
    x0 = np.double(1.0)
    base = np.double(0.5)

    for n in range(41):
        h = np.power(base, n, dtype=np.double)
        print(f"h: {h} -> f'(x): ", f_(x0, h))
        # file.write(str(h) + " " + str(f_(x0, h)) + "\n")

    print("\nLong double precision f'(x) value:\n")
    x0 = np.longdouble(1.0)
    base = np.longdouble(0.5)

    for n in range(41):
        h = np.power(base, n, dtype=np.longdouble)
        print(f"h: {h} -> f'(x): ", f_(x0, h))
        # file.write(str(h) + " " + str(f_(x0, h)) + "\n")

    print("\nExact value of f'(x):", f_exact(x0))

    print("\nSingle precision 1+h value:\n")
    base = np.single(0.5)

    for n in range(41):
        h = np.power(base, n, dtype=np.single)
        print(f"n: {n}, h: {h} -> 1+h: {1+h}")
        # file.write(str(n) + " " + str(h) + " " + str(1+h) + "\n")

    print("\nDouble precision 1+h value:\n")
    base = np.double(0.5)

    for n in range(41):
        h = np.power(base, n, dtype=np.double)
        print(f"n: {n}, h: {h} -> 1+h: {1+h}")
        # file.write(str(n) + " " + str(h) + " " + str(1+h) + "\n")

    print("\nLong double precision 1+h value:\n")
    base = np.longdouble(0.5)

    for n in range(41):
        h = np.power(base, n, dtype=np.longdouble)
        print(f"n: {n}, h: {h} -> 1+h: {1+h}")
        # file.write(str(n) + " " + str(h) + " " + str(1+h) + "\n")

    # file.close()

