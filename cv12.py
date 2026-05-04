import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

GRID_SIZE = 100
DENSITY = 0.5  # pocatecni hustota lesa
P_GROW = 0.05  # pravdepodobnost narustu noveho stromu
P_FIRE = 0.001  # pravdepodobnost samovzniceni (blesk)
INTERVAL_MS = 80

# stavy bunky
EMPTY = 0
TREE = 1
BURNING = 2
BURNT = 3

# barvy: prazdno = hneda, strom = zelena, hori = oranzova, spaleny = cerna
CMAP = ListedColormap(["#8B5A2B", "#118c1b", "#ff8800", "#000000"])

rng = np.random.default_rng()


def init_grid():
    grid = np.where(rng.random((GRID_SIZE, GRID_SIZE)) < DENSITY, TREE, EMPTY)
    return grid


def step(grid):
    # Posuneme grid do 4 smeru a zjistime, kde je horici soused (von Neumann).
    burning = grid == BURNING
    neighbor_fire = np.zeros_like(burning)
    neighbor_fire[1:, :] |= burning[:-1, :]
    neighbor_fire[:-1, :] |= burning[1:, :]
    neighbor_fire[:, 1:] |= burning[:, :-1]
    neighbor_fire[:, :-1] |= burning[:, 1:]

    new_grid = grid.copy()

    # 4. horici strom -> spaleny
    new_grid[grid == BURNING] = BURNT

    # 1. prazdno nebo spaleny -> strom s pravdepodobnosti p
    can_grow = (grid == EMPTY) | (grid == BURNT)
    grow = can_grow & (rng.random(grid.shape) < P_GROW)
    new_grid[grow] = TREE

    # 2. strom se zapalenym sousedem -> hori
    trees = grid == TREE
    catch = trees & neighbor_fire
    new_grid[catch] = BURNING

    # 3. strom bez horiciho souseda -> hori s pravdepodobnosti f (blesk)
    lightning = trees & ~neighbor_fire & (rng.random(grid.shape) < P_FIRE)
    new_grid[lightning] = BURNING

    return new_grid


def main():
    grid = init_grid()

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title("Forest Fire")
    ax.set_title("Forest Fire")
    image = ax.imshow(grid, cmap=CMAP, vmin=0, vmax=3, origin="lower")

    def update(_frame):
        nonlocal grid
        grid = step(grid)
        image.set_data(grid)
        return (image,)

    # cache_frame_data=False protoze animace bezi donekonecna
    _anim = FuncAnimation(
        fig, update, interval=INTERVAL_MS, blit=True, cache_frame_data=False
    )
    plt.show()


if __name__ == "__main__":
    main()
