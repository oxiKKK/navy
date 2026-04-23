from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUTPUT_PATH = "images/fractal_landscape.png"
RANDOM_SEED = 9
ITERATIONS = 10
IMAGE_WIDTH = 14
IMAGE_HEIGHT = 8


@dataclass(frozen=True)
class TerrainLayer:
    name: str
    start_y: float
    end_y: float
    initial_offset: float
    roughness: float
    min_height: float
    max_height: float
    color: str


LAYERS = (
    TerrainLayer(
        name="Puda",
        start_y=0.18,
        end_y=0.14,
        initial_offset=0.025,
        roughness=0.55,
        min_height=0.09,
        max_height=0.24,
        color="#8f4e00",
    ),
    TerrainLayer(
        name="Skaly",
        start_y=0.34,
        end_y=0.30,
        initial_offset=0.040,
        roughness=0.57,
        min_height=0.27,
        max_height=0.43,
        color="#101010",
    ),
    TerrainLayer(
        name="Vegetace",
        start_y=0.70,
        end_y=0.48,
        initial_offset=0.085,
        roughness=0.60,
        min_height=0.52,
        max_height=0.82,
        color="#118c1b",
    ),
)


def midpoint_displacement(start_y, end_y, iterations, initial_offset, roughness, rng):
    heights = np.array([start_y, end_y], dtype=float)
    offset = initial_offset

    for _ in range(iterations):
        refined = np.empty(heights.size * 2 - 1, dtype=float)
        refined[0::2] = heights

        # perbutace = nahodny posun
        perbutation = rng.uniform(-offset, offset, size=heights.size - 1)

        # vypocet noveho bodu jako prumer sousednich + perbutace
        refined[1::2] = ((heights[1:] + heights[:-1]) / 2) + perbutation
        heights = refined
        offset *= roughness

    x_axis = np.linspace(0.0, 1.0, heights.size)
    return x_axis, heights


def generate_layer_profile(layer, rng):
    x_axis, heights = midpoint_displacement(
        layer.start_y,
        layer.end_y,
        ITERATIONS,
        layer.initial_offset,
        layer.roughness,
        rng,
    )
    return x_axis, np.clip(heights, layer.min_height, layer.max_height)


def build_landscape(seed):
    rng = np.random.default_rng(seed)
    x_axis, soil = generate_layer_profile(LAYERS[0], rng)
    _, rocks = generate_layer_profile(LAYERS[1], rng)
    _, vegetation = generate_layer_profile(LAYERS[2], rng)

    rocks = np.maximum(rocks, soil + 0.08)
    vegetation = np.maximum(vegetation, rocks + 0.12)

    return x_axis, soil, rocks, vegetation


def draw_sky(ax):
    sky = np.linspace(0.0, 1.0, 512).reshape(512, 1)
    sky_cmap = LinearSegmentedColormap.from_list(
        "sky",
        ["#d9f0ff", "#b8e0ff", "#eef9ff"],
    )
    ax.imshow(
        sky,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        cmap=sky_cmap,
        aspect="auto",
        zorder=0,
    )


def save_figure():
    x_axis, soil, rocks, vegetation = build_landscape(RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(IMAGE_WIDTH, IMAGE_HEIGHT), constrained_layout=True)
    draw_sky(ax)

    ax.fill_between(x_axis, 0.0, soil, color=LAYERS[0].color, zorder=2)
    ax.fill_between(x_axis, soil, rocks, color=LAYERS[1].color, zorder=3)
    ax.fill_between(x_axis, rocks, vegetation, color=LAYERS[2].color, zorder=4)

    ax.plot(x_axis, soil, color="#6f3b00", linewidth=1.4, alpha=0.8, zorder=5)
    ax.plot(x_axis, rocks, color="#000000", linewidth=1.6, alpha=0.9, zorder=6)
    ax.plot(x_axis, vegetation, color="#0b6f13", linewidth=1.2, alpha=0.9, zorder=7)

    ax.set_title("CV9 - 2D krajina generovana midpoint displacement", fontsize=16)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    save_figure()
    print(f"Obrazek ulozen do: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
