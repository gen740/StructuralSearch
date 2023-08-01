import itertools
import subprocess

import matplotlib.pyplot as plt
import numpy as np


# capture subprocess stdout
def run_command(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if p.stdout is None or p.stderr is None:
        exit(1)

    for i in iter(p.stderr.readline, b""):
        print(i.decode(), end="")

    return iter(p.stdout.readline, b"")


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


if __name__ == "__main__":
    data = []
    for i in run_command(["./build/lattice_SA"]):
        data.append(np.array([float(x) for x in i.split()]))
    data = np.array(data)
    print(data)

    # plot N * 3 data to the 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d", aspect="equal", proj_type="ortho")

    for i in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
        ax.scatter(
            data[3:, 0] + i[0] * data[0, 0] + i[1] * data[1, 0] + i[2] * data[2, 0],
            data[3:, 1] + i[0] * data[0, 1] + i[1] * data[1, 1] + i[2] * data[2, 1],
            data[3:, 2] + i[0] * data[0, 2] + i[1] * data[1, 2] + i[2] * data[2, 2],
            # c=((i[0] + 1) / 2, (i[1] + 1) / 2, (i[2] + 1) / 2),
            c="k",
            marker="o",
        )
    set_axes_equal(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
