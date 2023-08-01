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


if __name__ == "__main__":
    data = []
    for i in run_command(["./build/lattice_SA"]):
        data.append(np.array([float(x) for x in i.split()]))
    data = np.array(data)

    # plot N * 3 data to the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
        ax.scatter(
            data[:, 0] + i[0], data[:, 1] + i[1], data[:, 2] + i[2], c="r", marker="o"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
