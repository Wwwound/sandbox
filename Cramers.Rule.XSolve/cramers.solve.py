import numpy as np
import numpy.linalg


def det(a, x=1):
    al = len(a)
    if al > 1:
        sign = -1
        total = 0
        for i in range(al):
            m = []
            for j in range(1, al):
                t = []
                for k in range(al):
                    if k != i:
                        t.append(a[j][k])
                m.append(t)
            sign *= -1
            total += x * det(m, sign * a[0][i])
        return total
    else:
        return x * a[0][0]


def solve_cramer(x, y):
    """
    Solve a linear matrix equation with Cramers Rule
    :param x: A matrix coefs
    :param y: Dependent variable values
    :return: ndarray
    """
    det_x = det(x)

    if np.any(abs(det_x) < 1e-10):
        raise ArithmeticError("No solutions")

    det_xi = np.ones_like(y, dtype="float64")

    for i in range(x.shape[0]):
        xi = np.hstack([x[:, :i], y.reshape(-1, 1), x[:, i + 1:]])
        det_xi[i] = det(xi)

    return det_xi / det_x


def check(a, b):
    print("A=\n", a)
    print("b=\n", b)

    try:
        x_cramers = solve_cramer(a, b)
        x_np = numpy.linalg.solve(a, b)
        x_calc = numpy.linalg.inv(a).dot(b)

        print("X Cramers Rule: \n", x_cramers)
        print("X (numpy): \n", x_np)
        print("X = inv(A) * b: \n", x_calc)
    except ArithmeticError as e:
        print(e)


print("=====Example 1======")
a = np.array([[2, 5, 4], [1, 3, 2], [2, 10, 9]])
b = np.array([30, 150, 110])
check(a, b)

print("=====Example 2======")
a2 = np.array([[1, 3, 3, 10],
               [3, 5, 7, 12],
               [10, 8, 3, -1],
               [-4, 1, 3, 0]])
b2 = np.array([5, 5, 7, 4])
check(a2, b2)

print("=====Example 3======")
a3 = np.array([[1, 3, 3, 10],
               [2, 6, 6, 20],
               [10, 8, 3, -1],
               [-4, 1, 3, 0]])
b3 = np.array([5, 5, 7, 4])
check(a3, b3)

