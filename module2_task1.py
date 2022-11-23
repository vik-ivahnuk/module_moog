# variant 13
# Завдання 1: Для заданої множини контрольних точок на площині побудувати сплайн Безьє.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
import math as mt
import json


def basis_bernstein_polinomial(i, n, t):
    return (mt.factorial(n) / (mt.factorial(i) * mt.factorial(n - i))) * pow(t, i) * pow(1 - t, n - i)


def bezier_curve(arr, degree, delta_t=0.01):
    result = []
    t = 0.0
    while t <= 1.0:
        if t > 1:
            t = 1
        coordinates = [0, 0]
        for i in range(degree + 1):
            basis_func = basis_bernstein_polinomial(i, DEGREE, t)
            coordinates[0] += arr[i][0] * basis_func
            coordinates[1] += arr[i][1] * basis_func
        result.append(coordinates)
        t += delta_t
    return result


if __name__ == "__main__":

    points_x = []
    points_y = []
    points = []
    with open("13.json", "r") as f:
        points = json.loads(f.read())["curve"]
        for point in points:
            points_x.append(point[0])
            points_y.append(point[1])
    DEGREE = len(points_x) - 1  # DEGREE = 9

    lines = np.empty(0)
    for i in range(DEGREE):
        lines = np.append(lines, [points_x[i], points_y[i], points_x[i + 1], points_y[i + 1]])
    lines = np.reshape(lines, (lines.shape[0] // 4, 2, 2))

    fig = plt.figure(1)
    ax = plt.axes()
    lc = mc.LineCollection(lines, colors='green')
    ax.add_collection(lc)

    bezier_points_x = np.empty(0)
    bezier_points_y = np.empty(0)
    ax.plot(points_x, points_y, 'ro')
    res = bezier_curve(points, DEGREE)

    for point in res:
        bezier_points_x = np.append(bezier_points_x, point[0])
        bezier_points_y = np.append(bezier_points_y, point[1])
    fig, = ax.plot(bezier_points_x, bezier_points_y, color='blue', linewidth=1.5)
    plt.savefig('result_1.jpg')

