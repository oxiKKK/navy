import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.5
EPOCHS = 20_000
RANDOM_SEED = 42
LOSS_PLOT_PATH = "images/xor_loss.png"


class NeuralNetworkXOR:
    def __init__(self, learning_rate, seed):
        rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate

        # Vstup obsahuje i bias, proto maji vahy hidden vrstvy rozmer 3x2.
        self.weights_hidden = rng.uniform(-1, 1, (3, 2))
        # Do vystupni vrstvy vstupuji 2 hidden neurony a bias.
        self.weights_output = rng.uniform(-1, 1, (3, 1))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(output):
        return output * (1.0 - output)

    def forward(self, inputs):
        # Forward pass: z vstupu spocitame aktivace hidden vrstvy a finalni vystup.
        hidden_net = inputs @ self.weights_hidden
        hidden_output = self.sigmoid(hidden_net)

        hidden_with_bias = np.concatenate(
            [hidden_output, np.ones((hidden_output.shape[0], 1))], axis=1
        )
        output_net = hidden_with_bias @ self.weights_output
        output = self.sigmoid(output_net)

        return hidden_output, hidden_with_bias, output

    def train(self, inputs, targets, epochs):
        loss_history = []

        for epoch in range(epochs):
            hidden_output, hidden_with_bias, output = self.forward(inputs)

            error = targets - output
            loss = np.mean(0.5 * np.square(error))
            loss_history.append(loss)

            # Backpropagation: nejdriv gradient na vystupu, potom preneseni chyby do hidden vrstvy.
            delta_output = error * self.sigmoid_derivative(output)
            hidden_error = delta_output @ self.weights_output[:-1].T
            delta_hidden = hidden_error * self.sigmoid_derivative(hidden_output)

            # Gradientni krok pro obe vrstvy.
            self.weights_output += (
                self.learning_rate * hidden_with_bias.T @ delta_output
            )
            self.weights_hidden += self.learning_rate * inputs.T @ delta_hidden

            if (epoch + 1) % 2_000 == 0:
                print(f"Epocha {epoch + 1:5d} | loss = {loss:.6f}")

        return loss_history

    def predict(self, inputs):
        _, _, output = self.forward(inputs)
        return output


def plot_loss(loss_history):
    plt.figure(figsize=(8, 4.5))
    plt.plot(loss_history, color="darkred", linewidth=2)
    plt.title("Prubeh treninkove chyby pro XOR")
    plt.xlabel("Epocha")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()


def main():
    # XOR vstupy ve formatu [x1, x2, bias]
    inputs = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    targets = np.array([[0], [1], [1], [0]], dtype=float)

    print("=== TRENINK JEDNODUCHE NEURONOVE SITE PRO XOR ===\n")
    network = NeuralNetworkXOR(learning_rate=LEARNING_RATE, seed=RANDOM_SEED)
    loss_history = network.train(inputs, targets, EPOCHS)

    predictions = network.predict(inputs)
    predicted_classes = (predictions >= 0.5).astype(int)
    accuracy = np.mean(predicted_classes == targets) * 100

    print("\n=== VYSLEDKY ===")
    for input, expected, predicted, predicted_class in zip(
        inputs[:, :2].astype(int), targets.astype(int), predictions, predicted_classes
    ):
        print(
            f"Vstup: {input.tolist()} | ocekavano: {expected[0]} | "
            f"vystup site: {predicted[0]:.6f} | trida: {predicted_class[0]}"
        )

    print(f"\nPresnost: {accuracy:.2f}%")
    print(f"Finalni loss: {loss_history[-1]:.8f}")

    print("\n=== VAHY ===")
    print("Hidden vrstva:")
    print(network.weights_hidden)
    print("\nOutput vrstva:")
    print(network.weights_output)
    print(f"\nGraf lossu ulozen do: {LOSS_PLOT_PATH}")

    plot_loss(loss_history)


if __name__ == "__main__":
    main()
