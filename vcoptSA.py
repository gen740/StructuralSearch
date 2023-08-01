import numpy as np
from numpy.typing import NDArray
from typing import cast
import itertools
import numpy.random as nr
import matplotlib.pyplot as plt
import math
from vcopt import vcopt


def t1():
    def otsuri(para):
        money = sum(para)
        return 592 - money

    para_range = [
        [0, 500],
        [0, 100, 200, 300, 400],
        [0, 50],
        [0, 10, 20, 30, 40],
        [0, 5],
        [0, 1, 2, 3, 4],
    ]

    v = vcopt()
    para, score = v.dcGA(para_range, otsuri, 0)  # パラメータ範囲  # 評価関数  # 目標値


def t2(seed):
    nr.seed(seed)

    town_x = nr.rand(10)
    town_y = nr.rand(10)

    def tsp_score(para):
        return np.sum(
            np.hypot(
                town_x[para][:-1] - town_x[para][1:],
                town_y[para][:-1] - town_y[para][1:],
            )
        )

    para = range(10)
    # para, score = vcopt().opt2(para, tsp_score, 0.0)
    para, score = vcopt().tspGA(para, tsp_score, 0.0)
    print(para, score)

    plt.plot(town_x[para], town_y[para], "ok-")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


def t3():
    nr.seed(0)

    DIM = 4

    candidate = []
    for i in np.linspace(0, 1, 101):
        for j in np.linspace(0, 1, 101):
            for k in np.linspace(0, 1, 101):
                candidate.append([i, j, k])
    candidate = np.array(candidate)

    points = np.array([candidate for _ in range(DIM)])
    print(points.shape)

    para_range = [[i for i in range(len(candidate))] for _ in range(DIM)]

    def tsp_score(para):
        ret = 0
        for i in range(len(para)):
            value_min = []
            for j in range(len(para)):
                if i != j:
                    dest = [0, 0, 0]
                    for k in range(3):
                        dest[k] = abs(points[i, para[i], k] - points[j, para[j], k])
                        if dest[k] >= 0.5:
                            dest[k] = 1 - dest[k]
                    d = math.hypot(*[dest[k] for k in range(3)])
                    if d == 0:
                        value_min.append(0.0000001)
                    else:
                        value_min.append(d)
            value_min.sort()
            for k in range(DIM - 1):
                ret += 1 / (value_min[k] ** 6)
                # ret += value_min[k] ** 1
        return ret

    para, score = vcopt().dcGA(para_range, tsp_score, 0)
    print(para, score)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax = fig.add_subplot(111)

    # for i in range(27):
    #     for j in range(len(para)):
    #         # print(points[j, para[j], 0] + i % 3 - 1, points[j, para[j], 1] + i // 3 - 1)
    #         ax.plot(
    #             points[j, para[j], 0] + i % 3 - 1,
    #             points[j, para[j], 1] + (i // 3) % 3 - 1,
    #             points[j, para[j], 2] + (i // 3) // 3 - 1,
    #             ".k",
    #         )
    for j in range(len(para)):
        ax.plot(
            points[j, para[j], 0],
            points[j, para[j], 1],
            points[j, para[j], 2],
            ".k",
        )
    # plt.set_figsize(2, 2)
    plt.grid()

    plt.show()


def energy(p: list[NDArray]) -> float:
    ret = 0
    for i in range(len(p)):
        for j in range(i, len(p)):
            if i == j:
                continue
            dist: float = 0.0
            for k in itertools.product(
                [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]
            ):
                dist = cast(float, np.linalg.norm(p[i] - p[j] + np.array(k)))
                if dist == 0:
                    ret += 1 << 30
                else:
                    ret += 1 / (dist**6)
    return ret


def t4():
    nr.seed(0)

    NUM = 10
    DIM = 3

    candidate = np.linspace(0, 1, 101)
    # candidate = np.random.rand(100)

    points = np.array([candidate for _ in range(NUM * DIM)])
    print(points.shape)

    def tsp_score(para):
        p = []
        for i in range(len(para) // DIM):
            p.append(np.array([para[3 * i], para[3 * i + 1], para[3 * i + 2]]))
        return energy(p)

    para, score = vcopt().dcGA(points, tsp_score, 0)
    print(para, score)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(27):
        for j in range(len(para) // DIM):
            ax.plot(
                para[3 * j] + i % 3 - 1,
                para[3 * j + 1] + (i // 3) % 3 - 1,
                para[3 * j + 2] + (i // 3) // 3 - 1,
                ".k",
            )
    plt.grid()

    plt.show()


def energy2D(p: list[NDArray]) -> float:
    ret = 0
    for i in range(len(p)):
        for j in range(i, len(p)):
            dist: float = 0.0

            # d is normalized vector of nearest distance
            d = p[i] - p[j]
            d[0] = np.abs(d[0])
            d[1] = np.abs(d[1])

            if d[0] >= 0.5:
                d[0] = 1 - d[0]
            if d[1] >= 0.5:
                d[1] = 1 - d[1]

            for k in itertools.product(
                [0.0, 1.0],
                [0.0, 1.0],
            ):
                if i == j and k == (0, 0):
                    continue
                dist = cast(float, np.linalg.norm(d - np.array(k)))
                if dist == 0:
                    ret += 1 << 30
                else:
                    ret += 1 / (dist**6)
    return ret


def t42D():
    nr.seed(0)

    NUM = 2
    DIM = 2

    points = np.array([nr.rand(16) for _ in range(NUM * DIM)])

    def tsp_score(para):
        p = []
        for i in range(len(para) // DIM):
            p.append(np.array([para[DIM * i], para[DIM * i + 1]]))
        return energy2D(p)

    para, score = vcopt().dcGA(points, tsp_score, 0)
    print(para, score)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in itertools.product([-1, 0, 1], [-1, 0, 1]):
        for j in range(len(para) // DIM):
            ax.plot(
                para[DIM * j] + i[0],
                para[DIM * j + 1] + i[1],
                "ok",
            )
    plt.grid()

    plt.show()


if __name__ == "__main__":
    t42D()
