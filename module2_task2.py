# variant 13
# Завдання 2: Для заданої множини контрольних точок у просторі (13x13 points) побудувати поверхню NURBS

import numpy as np
from tqdm import tqdm
from functools import lru_cache
from matplotlib.tri.triangulation import Triangulation
import matplotlib.pyplot as plt
import json

DEGREE = 3


def N_i_k(i, k, T, t):
    if k == 1:
        if T[i] <= t < T[i + 1]:
            return 1
        return 0
    return (t - T[i]) / (T[i + k - 1] - T[i]) * N_i_k(i, k - 1, T, t) + \
           (T[i + k] - t) / (T[i + k] - T[i + 1]) * N_i_k(i + 1, k - 1, T, t)


@lru_cache(maxsize=128)
def all_N_i_k(k, U, u, n):
    return [N_i_k(i, k, U, u) for i in range(n)]


def calc_R(k, l, dim, U, u, v):
    all_N_u = all_N_i_k(DEGREE, U, u, dim[0])
    all_N_v = all_N_i_k(DEGREE, U, v, dim[1])
    num_ = all_N_u[k] * all_N_v[l]
    if not num_:
        return num_
    denum_ = 0
    for p in range(dim[0]):
        for q in range(dim[1]):
            denum_ += all_N_u[p] * all_N_v[q]
    return num_ / denum_


def nurbs_surface(points, dim, indices):
    U = tuple(range(dim[0] + DEGREE + 1))
    controls = np.arange(max(U), step=0.1)
    result = []
    for u in tqdm(controls):
        for v in controls:
            coordinates = [0, 0, 0]
            for k in range(dim[0]):
                for l in range(dim[1]):
                    p_ = points[indices.index([k, l])]
                    r_ = calc_R(k, l, dim, U, u, v)
                    coordinates[0] += p_[0] * r_
                    coordinates[1] += p_[1] * r_
                    coordinates[2] += p_[2] * r_
            if coordinates != [0, 0, 0]:
                result.append(coordinates)
    return result


def save_to_file(points):
    points = np.array(points)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    triangulation = Triangulation(points[:, 0], points[:, 1])
    ax.plot_trisurf(triangulation, points[:, 2])
    fig.savefig('result_2.jpg')


if __name__ == "__main__":
    with open('13.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        dim = data['surface']['gridSize']
        indices = data['surface']['indices']
        data = data['surface']['points']
        points = [p for p in data]
    nurbs = nurbs_surface(points, dim, indices)
    save_to_file(nurbs)
