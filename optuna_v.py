import copy
import itertools
import logging
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import optuna
from mpl_toolkits.mplot3d import proj3d
from numpy.typing import NDArray
from optuna import Trial
from optuna.samplers import CmaEsSampler, RandomSampler


def orthogonal_transformation(zfront, zback, _):
    a = 2 / (zfront - zback)
    b = -1 * (zfront + zback) / (zfront - zback)
    c = zback
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, a, b], [0, 0, 0, c]])


proj3d.persp_transformation = orthogonal_transformation

optuna.logging.set_verbosity(logging.WARN)

NUM = 3


def energy(p: list[NDArray], x1: NDArray, x2: NDArray, x3: NDArray) -> float:
    ret = 0
    for i in range(NUM):
        for j in range(i, NUM):
            dist: float = 0.0
            for k in itertools.product(
                [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            ):
                if i == j and k == (0, 0, 0):
                    continue
                dist = cast(
                    float,
                    np.linalg.norm(p[j] + x1 * k[0] + x2 * k[1] + x3 * k[2] - p[i]),
                )
                if dist == 0:
                    ret += 1 << 30
                else:
                    ret += 1 / (dist**12) - 1 / (dist**6)
    return ret


def objective(trial: Trial):
    RMAX = 10.0
    RMIN = 0.2
    p = []

    # create det = 1 vector
    r1 = trial.suggest_float(f"a_1", RMIN, RMAX)

    # create vector
    x1 = np.array(
        [
            r1,
            0,
            0,
        ]
    )

    # r2 = trial.suggest_float(f"a_2", RMIN, RMAX)
    r2 = r1
    # phi2 = trial.suggest_float(f"phi_2", np.pi / 6, np.pi * 5 / 6)
    phi2 = np.pi / 3

    # create vector
    x2 = np.array([r2 * np.cos(phi2), r2 * np.sin(phi2), 0])

    r3 = trial.suggest_float(f"a_3", RMIN, RMAX)
    th3 = trial.suggest_float(f"th_3", np.pi / 6, np.pi / 2)
    phi3 = trial.suggest_float(f"phi_3", 0.0, 2 * np.pi)
    x3 = np.array(
        [
            r3 * np.cos(th3) * np.cos(phi3),
            r3 * np.cos(th3) * np.sin(phi3),
            r3 * np.sin(th3),
        ]
    )

    P = copy.deepcopy(np.array([x1, x2, x3]))

    trial.set_user_attr("x11", x1[0])
    trial.set_user_attr("x12", x1[1])
    trial.set_user_attr("x13", x1[2])
    trial.set_user_attr("x21", x2[0])
    trial.set_user_attr("x22", x2[1])
    trial.set_user_attr("x23", x2[2])
    trial.set_user_attr("x31", x3[0])
    trial.set_user_attr("x32", x3[1])
    trial.set_user_attr("x33", x3[2])

    for i in range(NUM):
        point = P.T @ np.array(
            [
                trial.suggest_float(f"x_{i}", 0.0, 1.0),
                trial.suggest_float(f"y_{i}", 0.0, 1.0),
                trial.suggest_float(f"z_{i}", 0.0, 1.0),
            ]
        )
        trial.set_user_attr(f"p{i}_x", point[0])
        trial.set_user_attr(f"p{i}_y", point[1])
        trial.set_user_attr(f"p{i}_z", point[2])

        p.append(point)
    return energy(p, x1, x2, x3)


# [[ 1.74333026e+00  0.00000000e+00  0.00000000e+00]
#  [-2.61609941e-05  1.10290510e+00  0.00000000e+00]
#  [-2.61628917e-05  5.48823973e-01  9.56656635e-01]]
# 0.868123275432113, 0.9861329031200726, 0.943976395589311
# 0.36813507110873167, 0.6525818720178835, 0.6104253524965662

study = optuna.create_study(
    direction="minimize",
    sampler=CmaEsSampler(),
)
# study = optuna.create_study(direction="minimize", sampler=RandomSampler())
# study = optuna.create_study(direction="minimize")

study.optimize(objective, n_trials=20_000, show_progress_bar=True)
print(study.best_trial)

x1 = np.array(  # type: ignore
    [
        study.best_trial.user_attrs["x11"],
        study.best_trial.user_attrs["x12"],
        study.best_trial.user_attrs["x13"],
    ]
)

x2 = np.array(  # type: ignore
    [
        study.best_trial.user_attrs["x21"],
        study.best_trial.user_attrs["x22"],
        study.best_trial.user_attrs["x23"],
    ]
)

x3 = np.array(  # type: ignore
    [
        study.best_trial.user_attrs["x31"],
        study.best_trial.user_attrs["x32"],
        study.best_trial.user_attrs["x33"],
    ]
)

P = np.array([x1, x2, x3])


print(P)
for i in range(NUM):
    print(
        f'{study.best_params[f"x_{i}"]}, {study.best_params[f"y_{i}"]}, {study.best_params[f"z_{i}"]}'
    )


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    lack of an aspect='equal' setting for 3D plots as of matplotlib 2.1.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call axes.ellipse.
    radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121, projection="3d")
ax1.view_init(elev=30, azim=45)
ax2 = fig.add_subplot(122, projection="3d")

# summary
for j in itertools.product([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]):
    for k in range(NUM):
        point = np.array(
            [
                study.best_trial.user_attrs[f"p{k}_x"],
                study.best_trial.user_attrs[f"p{k}_y"],
                study.best_trial.user_attrs[f"p{k}_z"],
            ]
        )
        point += j[0] * x1 + j[1] * x2 + j[2] * x3
        ax1.plot(point[0], point[1], point[2], "o", color=tuple((l + 1) / 3 for l in j))

# unit cell
ax2.plot([0, x1[0]], [0, x1[1]], [0, x1[2]], color=(0.5, 0.5, 0.5))
ax2.plot([0, x2[0]], [0, x2[1]], [0, x2[2]], color=(0.5, 0.5, 0.5))
ax2.plot([0, x3[0]], [0, x3[1]], [0, x3[2]], color=(0.5, 0.5, 0.5))
ax2.plot(
    [x1[0], x1[0] + x2[0]],
    [x1[1], x1[1] + x2[1]],
    [x1[2], x1[2] + x2[2]],
    color=(0.5, 0.5, 0.5),
)
ax2.plot(
    [x1[0], x1[0] + x3[0]],
    [x1[1], x1[1] + x3[1]],
    [x1[2], x1[2] + x3[2]],
    color=(0.5, 0.5, 0.5),
)
ax2.plot(
    [x2[0], x2[0] + x1[0]],
    [x2[1], x2[1] + x1[1]],
    [x2[2], x2[2] + x1[2]],
    color=(0.5, 0.5, 0.5),
)
ax2.plot(
    [x2[0], x2[0] + x3[0]],
    [x2[1], x2[1] + x3[1]],
    [x2[2], x2[2] + x3[2]],
    color=(0.5, 0.5, 0.5),
)
ax2.plot(
    [x3[0], x3[0] + x1[0]],
    [x3[1], x3[1] + x1[1]],
    [x3[2], x3[2] + x1[2]],
    color=(0.5, 0.5, 0.5),
)
ax2.plot(
    [x3[0], x3[0] + x2[0]],
    [x3[1], x3[1] + x2[1]],
    [x3[2], x3[2] + x2[2]],
    color=(0.5, 0.5, 0.5),
)
ax2.plot(
    [x1[0] + x2[0], x1[0] + x2[0] + x3[0]],
    [x1[1] + x2[1], x1[1] + x2[1] + x3[1]],
    [x1[2] + x2[2], x1[2] + x2[2] + x3[2]],
    color=(0.5, 0.5, 0.5),
)
ax2.plot(
    [x1[0] + x3[0], x1[0] + x3[0] + x2[0]],
    [x1[1] + x3[1], x1[1] + x3[1] + x2[1]],
    [x1[2] + x3[2], x1[2] + x3[2] + x2[2]],
    color=(0.5, 0.5, 0.5),
)
ax2.plot(
    [x2[0] + x3[0], x2[0] + x3[0] + x1[0]],
    [x2[1] + x3[1], x2[1] + x3[1] + x1[1]],
    [x2[2] + x3[2], x2[2] + x3[2] + x1[2]],
    color=(0.5, 0.5, 0.5),
)


# plot points
for i in range(NUM):
    point = np.array(
        [
            study.best_trial.user_attrs[f"p{i}_x"],
            study.best_trial.user_attrs[f"p{i}_y"],
            study.best_trial.user_attrs[f"p{i}_z"],
        ]
    )
    ax2.plot(point[0], point[1], point[2], "o", color="k")


set_axes_equal(ax1)
set_axes_equal(ax2)

plt.grid()

plt.show()
