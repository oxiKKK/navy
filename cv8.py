from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle

OUTPUT_PATH = "images/mandelbrot_zoom.png"
IMAGE_WIDTH = 1400
IMAGE_HEIGHT = 900


@dataclass(frozen=True)
class Viewport:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    max_iterations: int


FULL_VIEW = Viewport(
    name="Cely Mandelbrotuv set",
    xmin=-2.0,
    xmax=1.0,
    ymin=-1.5,
    ymax=1.5,
    max_iterations=200,
)

ZOOM_VIEW = Viewport(
    name="Zoom: Seahorse Valley",
    xmin=-0.755,
    xmax=-0.735,
    ymin=0.105,
    ymax=0.125,
    max_iterations=350,
)


def compute_mandelbrot(view, width, height):
    real_axis = np.linspace(view.xmin, view.xmax, width)
    imaginary_axis = np.linspace(view.ymin, view.ymax, height)
    complex_plane = real_axis[np.newaxis, :] + 1j * imaginary_axis[:, np.newaxis]

    # z <- z^2 + c, kde c je bod z komplexni roviny a z zacina na 0

    z_values = np.zeros_like(complex_plane)
    escape_iterations = np.full(complex_plane.shape, view.max_iterations, dtype=int)
    active_mask = np.ones(complex_plane.shape, dtype=bool)

    for iteration in range(view.max_iterations):
        z_values[active_mask] = (
            z_values[active_mask] * z_values[active_mask] + complex_plane[active_mask]
        )
        escaped = np.greater(np.abs(z_values), 2.0, where=active_mask)
        just_escaped = escaped & active_mask
        escape_iterations[just_escaped] = iteration
        active_mask[just_escaped] = False

        if not np.any(active_mask):
            break

    return escape_iterations


def build_colored_image(iterations, max_iterations):
    normalized = iterations / max_iterations
    hue = (0.95 + 8.0 * normalized) % 1.0
    saturation = np.where(iterations < max_iterations, 0.85, 0.0)
    value = np.where(iterations < max_iterations, 1.0, 0.0)

    hsv_image = np.dstack((hue, saturation, value))
    return hsv_to_rgb(hsv_image)


def render_view(ax, view, width, height):
    iterations = compute_mandelbrot(view, width, height)
    image = build_colored_image(iterations, view.max_iterations)

    ax.imshow(
        image,
        extent=(view.xmin, view.xmax, view.ymin, view.ymax),
        origin="lower",
        interpolation="bilinear",
    )
    ax.set_title(f"{view.name}\nmax iteraci = {view.max_iterations}")
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    ax.set_aspect("equal")


def save_figure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle("CV8 - Mandelbrotuv set a detailni zoom", fontsize=16)

    render_view(axes[0], FULL_VIEW, IMAGE_WIDTH, IMAGE_HEIGHT)
    render_view(axes[1], ZOOM_VIEW, IMAGE_WIDTH, IMAGE_HEIGHT)

    zoom_outline = Rectangle(
        (ZOOM_VIEW.xmin, ZOOM_VIEW.ymin),
        ZOOM_VIEW.xmax - ZOOM_VIEW.xmin,
        ZOOM_VIEW.ymax - ZOOM_VIEW.ymin,
        linewidth=1.5,
        edgecolor="white",
        facecolor="none",
    )
    axes[0].add_patch(zoom_outline)

    plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    save_figure()
    print(f"Obrazek ulozen do: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
