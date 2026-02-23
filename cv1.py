import random
import plotly.graph_objects as go
import plotly.io as pio

# Rovnice primky: y = 3x + 2
M = 3
B = 2
POCET_BODU = 100
MIN_HODNOTA = -10
MAX_HODNOTA = 10


def perceptron_klasifikace(x, y):
    # Jednoduchy perceptron pro rozhodovaci hranici -3x + y - 2 = 0
    # > 0 ... bod je nad primkou, < 0 ... pod primkou, = 0 ... na primce
    net = (-M * x) + y - B
    if net > 0:
        return "nad"
    if net < 0:
        return "pod"
    return "na"


def main():
    # Vygenerujeme 100 bodu s celymi souradnicemi
    body = [
        (
            random.randint(MIN_HODNOTA, MAX_HODNOTA),
            random.randint(MIN_HODNOTA, MAX_HODNOTA),
        )
        for _ in range(POCET_BODU)
    ]

    nad_x, nad_y = [], []
    pod_x, pod_y = [], []
    na_x, na_y = [], []

    # Klasifikace bodu perceptronem
    for x, y in body:
        trida = perceptron_klasifikace(x, y)
        if trida == "nad":
            nad_x.append(x)
            nad_y.append(y)
        elif trida == "pod":
            pod_x.append(x)
            pod_y.append(y)
        else:
            na_x.append(x)
            na_y.append(y)

    # Vykresleni primky
    x1 = MIN_HODNOTA - 1
    x2 = MAX_HODNOTA + 1
    y1 = M * x1 + B
    y2 = M * x2 + B

    fig = go.Figure()

    # Primka y = 3x + 2
    fig.add_trace(
        go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            name="y = 3x + 2",
            line=dict(color="black", width=2),
        )
    )

    # Body podle tridy
    fig.add_trace(
        go.Scatter(
            x=nad_x,
            y=nad_y,
            mode="markers",
            name="Nad primkou",
            marker=dict(color="red", size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pod_x,
            y=pod_y,
            mode="markers",
            name="Pod primkou",
            marker=dict(color="blue", size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=na_x,
            y=na_y,
            mode="markers",
            name="Na primce",
            marker=dict(color="green", size=10),
        )
    )

    fig.update_layout(
        title="Task 1 - Perceptron: point on the line",
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(
            range=[MIN_HODNOTA - 1, MAX_HODNOTA + 1],
            zeroline=True,
            zerolinecolor="gray",
            showgrid=True,
        ),
        yaxis=dict(
            range=[MIN_HODNOTA - 1, MAX_HODNOTA + 1],
            zeroline=True,
            zerolinecolor="gray",
            showgrid=True,
        ),
        template="plotly_white",
    )
    pio.renderers.default = "browser"
    fig.show()


if __name__ == "__main__":
    main()
