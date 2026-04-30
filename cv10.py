import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "images/logistic_bifurcation.png"

TRANSIENT = 200
PLOT_ITERS = 100

HIDDEN = 32  # pocet neuronu ve skryte vrstve
LEARNING_RATE = 0.1
EPOCHS = 500

rng = np.random.default_rng(42)


def logistic_map(a, x):
    return a * x * (1.0 - x)


def bifurcation(a_values):
    # bifurkace = bod, kde se chovani systemu meni.

    x = np.full(a_values.size, 0.5)
    for _ in range(TRANSIENT):
        x = logistic_map(a_values, x)

    a_points, x_points = [], []
    for _ in range(PLOT_ITERS):
        x = logistic_map(a_values, x)
        a_points.append(a_values.copy())
        x_points.append(x.copy())
    return np.concatenate(a_points), np.concatenate(x_points)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(s):
    return s * (1.0 - s)


def train_network(a_data, x_data, target):
    # Vstup: [a/4, x, bias]
    # skryta vrstva tanh
    # vystup sigmoid

    # W1 = vahy mezi vstupem a skrytou siti
    # W2 = vahy mezi skrytou siti a vystupem
    W1 = rng.normal(0, 0.5, (3, HIDDEN))
    W2 = rng.normal(0, 0.5, (HIDDEN + 1, 1))

    inputs = np.column_stack([a_data / 4.0, x_data, np.ones_like(a_data)])
    targets = target[:, None]

    # pro kazdou epochu vypocitame vystup, spocitame error a upravime vahy
    # pomoci gradientniho sestupu
    for epoch in range(EPOCHS):
        h = np.tanh(inputs @ W1)  # vstup -> skryta vrstva
        h_bias = np.column_stack([h, np.ones(h.shape[0])])
        y = sigmoid(h_bias @ W2)  # skryta vrstva -> vystup

        # chyba = skutecna hodnota - predikce
        error = targets - y
        delta_y = error * sigmoid_derivative(y)  # vypocet gradientu
        delta_h = (delta_y @ W2[:-1].T) * (1.0 - h * h)

        # uprava vah pomoci gradientniho sestupu
        W2 += LEARNING_RATE * h_bias.T @ delta_y / inputs.shape[0]
        W1 += LEARNING_RATE * inputs.T @ delta_h / inputs.shape[0]

        if (epoch + 1) % 50 == 0:
            print(f"Epocha {epoch + 1} | loss = {np.mean(error**2):.5f} MSE")

    return W1, W2


def predict(W1, W2, a_data, x_data):
    # Vezmeme natrenovane vahy a pro vstupni data vypocitame predikci
    inputs = np.column_stack([a_data / 4.0, x_data, np.ones_like(a_data)])
    # vypocitame skrytou vrstvu
    h = np.tanh(inputs @ W1)
    # pridame bias pro skrytou vrstvu
    h_bias = np.column_stack([h, np.ones(h.shape[0])])
    # vypocitame vystup a vratime jako 1D pole
    return sigmoid(h_bias @ W2).flatten()


def main():
    a_values = np.linspace(0.0, 4.0, 10000)

    # Trenovaci data: dvojice (a, x_n) -> x_{n+1} ze skutecne logisticke mapy
    a_actual, x_actual = bifurcation(a_values)
    targets = logistic_map(a_actual, x_actual)

    print("Trenuji sit...")
    W1, W2 = train_network(a_actual, x_actual, targets)

    print("Predikuji...")
    x_predicted = predict(W1, W2, a_actual, x_actual)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Logisticka mapa")

    ax1.scatter(a_actual, x_actual, s=0.2, color="black")
    ax1.set_title("Skutecny bifurkacni diagram")
    ax1.set_xlabel("a")
    ax1.set_ylabel("x")

    ax2.scatter(a_actual, x_actual, s=0.2, color="black", label="skutecne")
    ax2.scatter(a_actual, x_predicted, s=0.4, color="red", label="predikce")
    ax2.set_title("Predikce neuronovou siti")
    ax2.set_xlabel("a")
    ax2.legend(markerscale=8)

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ulozeno: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
