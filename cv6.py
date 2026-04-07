from dataclasses import dataclass
from math import cos, pi, sin

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

OUTPUT_PATH = "images/lsystems.png"


@dataclass(frozen=True)
class LSystemSpec:
    name: str
    axiom: str
    rules: dict[str, str]
    angle: float
    iterations: int
    initial_angle: float = 0.0


SYSTEMS = [
    LSystemSpec(
        name="Dragon Curve",
        axiom="FX",
        rules={"X": "X+YF+", "Y": "-FX-Y"},
        angle=pi / 2,
        iterations=12,
    ),
    LSystemSpec(
        name="Gosper Curve",
        axiom="XF",
        rules={
            "X": "X+YF++YF-FX--FXFX-YF+",
            "Y": "-FX+YFYF++YF+FX--FX-Y",
        },
        angle=pi / 3,
        iterations=4,
    ),
    LSystemSpec(
        name="Sierpinski Triangle",
        axiom="FXF--FF--FF",
        rules={
            "F": "FF",
            "X": "--FXF++FXF++FXF--",
        },
        angle=pi / 3,
        iterations=5,
    ),
    LSystemSpec(
        name="Bush",
        axiom="X",
        rules={"F": "FF", "X": "F[+X]F[-X]+X"},
        angle=pi / 9,
        iterations=6,
        initial_angle=pi / 2,
    ),
]


def expand_lsystem(axiom, rules, iterations):
    sequence = axiom
    for _ in range(iterations):
        sequence = "".join(rules.get(symbol, symbol) for symbol in sequence)
    return sequence


def build_segments(commands, angle_step, step=1.0, initial_angle=0.0):
    x, y = 0.0, 0.0
    angle = initial_angle
    stack = []
    segments = []

    min_x = max_x = x
    min_y = max_y = y

    for command in commands:
        if command in {"F", "b"}:
            next_x = x + step * cos(angle)
            next_y = y + step * sin(angle)

            if command == "F":
                segments.append(((x, y), (next_x, next_y)))

            x, y = next_x, next_y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        elif command == "+":
            angle -= angle_step
        elif command == "-":
            angle += angle_step
        elif command == "[":
            stack.append((x, y, angle))
        elif command == "]":
            x, y, angle = stack.pop()

    bounds = (min_x, max_x, min_y, max_y)
    return segments, bounds


def render_system(ax, spec):
    commands = expand_lsystem(spec.axiom, spec.rules, spec.iterations)
    segments, bounds = build_segments(
        commands, spec.angle, initial_angle=spec.initial_angle
    )

    collection = LineCollection(segments, colors="darkgreen", linewidths=1.0)
    ax.add_collection(collection)

    min_x, max_x, min_y, max_y = bounds
    padding_x = max((max_x - min_x) * 0.05, 1.0)
    padding_y = max((max_y - min_y) * 0.05, 1.0)

    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{spec.name}\niterace={spec.iterations}, uhel={spec.angle:.3f} rad")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("CV6 - L-systemy z fedimser.github.io", fontsize=16)

    for ax, spec in zip(axes.flat, SYSTEMS):
        render_system(ax, spec)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Obrazek ulozen do: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
