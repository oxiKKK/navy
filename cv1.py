import random
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# Rovnice primky: y = 3x + 2
M = 3
B = 2
POCET_TRAIN = 80  # pocet trenovacich bodu
POCET_TEST = 20   # pocet testovacich bodu
MIN_HODNOTA = -10
MAX_HODNOTA = 10
LEARNING_RATE = 0.1
EPOCHS = 100


class Perceptron:
    def __init__(self, learning_rate=0.1):
        # Vahy: w1 pro x, w2 pro y, w0 pro bias
        # Inicializace vah nahodne mezi -1 a 1
        self.w = np.random.uniform(-1, 1, 3)
        self.learning_rate = learning_rate
        
    def predict(self, x):
        """Predikce pro vstupni vektor x (s biasem)"""
        net = np.dot(self.w, x)
        return 1 if net > 0 else -1
    
    def train(self, X, y, epochs):
        """Trenovani perceptronu"""
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                if error != 0:
                    # Aktualizace vah
                    self.w += self.learning_rate * error * xi
                    errors += 1
            if errors == 0:
                print(f"Konvergence dosazena v epoce {epoch + 1}")
                break
        print(f"Finalni vahy: w1={self.w[0]:.4f}, w2={self.w[1]:.4f}, bias={self.w[2]:.4f}")
    
    def evaluate(self, X, y):
        """Evaluace modelu na testovacich datech"""
        correct = 0
        for xi, yi in zip(X, y):
            if self.predict(xi) == yi:
                correct += 1
        accuracy = correct / len(y) * 100
        return accuracy, correct, len(y)


def generate_labeled_data(n):
    """Generuje body a jejich label (1 = nad primkou, -1 = pod primkou)"""
    data = []
    for _ in range(n):
        x = random.randint(MIN_HODNOTA, MAX_HODNOTA)
        y = random.randint(MIN_HODNOTA, MAX_HODNOTA)
        
        # Skutecna hranice: y = 3x + 2, tedy 3x - y + 2 = 0
        # Pokud y > 3x + 2 (3x - y + 2 < 0), bod je NAD primkou → label 1
        # Pokud y < 3x + 2 (3x - y + 2 > 0), bod je POD primkou → label -1
        value = M * x - y + B
        label = 1 if value < 0 else -1
        
        # Format: [x, y, bias=1]
        data.append(([x, y, 1], label))
    
    return data


def main():
    print("=== TRENINK PERCEPTRONU ===\n")
    
    # Generovani trenovacich a testovacich dat
    train_data = generate_labeled_data(POCET_TRAIN)
    test_data = generate_labeled_data(POCET_TEST)
    
    X_train = np.array([d[0] for d in train_data])
    y_train = np.array([d[1] for d in train_data])
    
    X_test = np.array([d[0] for d in test_data])
    y_test = np.array([d[1] for d in test_data])
    
    # Vytvoreni a trenovani perceptronu
    perceptron = Perceptron(learning_rate=LEARNING_RATE)
    print(f"Pocet trenovacich dat: {POCET_TRAIN}")
    print(f"Pocet testovacich dat: {POCET_TEST}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochy: {EPOCHS}\n")
    
    perceptron.train(X_train, y_train, EPOCHS)
    
    # Evaluace na trenovacich datech
    train_acc, train_correct, train_total = perceptron.evaluate(X_train, y_train)
    print(f"\n=== VYSLEDKY NA TRENOVACICH DATECH ===")
    print(f"Presnost: {train_acc:.2f}% ({train_correct}/{train_total})")
    
    # Evaluace na testovacich datech
    test_acc, test_correct, test_total = perceptron.evaluate(X_test, y_test)
    print(f"\n=== VYSLEDKY NA TESTOVACICH DATECH ===")
    print(f"Presnost: {test_acc:.2f}% ({test_correct}/{test_total})")
    
    # Vizualizace
    print("\n=== VIZUALIZACE ===")
    
    # Vykresleni skutecne primky
    x1 = MIN_HODNOTA - 1
    x2 = MAX_HODNOTA + 1
    y1 = M * x1 + B
    y2 = M * x2 + B
    
    fig = go.Figure()
    
    # Skutecna primka
    fig.add_trace(
        go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            name="Skutecna primka (y = 3x + 2)",
            line=dict(color="black", width=2),
        )
    )
    
    # Naucena primka z perceptronu
    # w1*x + w2*y + w0 = 0 => y = -(w1/w2)*x - (w0/w2)
    if perceptron.w[1] != 0:
        y1_pred = -(perceptron.w[0] / perceptron.w[1]) * x1 - (perceptron.w[2] / perceptron.w[1])
        y2_pred = -(perceptron.w[0] / perceptron.w[1]) * x2 - (perceptron.w[2] / perceptron.w[1])
        
        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1_pred, y2_pred],
                mode="lines",
                name="Naucena primka",
                line=dict(color="purple", width=2, dash="dash"),
            )
        )
    
    # Trenovaci body
    train_nad_x, train_nad_y = [], []
    train_pod_x, train_pod_y = [], []
    
    for (x, y, _), label in train_data:
        if label == 1:
            train_nad_x.append(x)
            train_nad_y.append(y)
        else:
            train_pod_x.append(x)
            train_pod_y.append(y)
    
    fig.add_trace(
        go.Scatter(
            x=train_nad_x,
            y=train_nad_y,
            mode="markers",
            name="Trenovaci: nad primkou",
            marker=dict(color="red", size=8, symbol="circle"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_pod_x,
            y=train_pod_y,
            mode="markers",
            name="Trenovaci: pod primkou",
            marker=dict(color="blue", size=8, symbol="circle"),
        )
    )
    
    # Testovaci body
    test_nad_x, test_nad_y = [], []
    test_pod_x, test_pod_y = [], []
    
    for (x, y, _), label in test_data:
        if label == 1:
            test_nad_x.append(x)
            test_nad_y.append(y)
        else:
            test_pod_x.append(x)
            test_pod_y.append(y)
    
    fig.add_trace(
        go.Scatter(
            x=test_nad_x,
            y=test_nad_y,
            mode="markers",
            name="Testovaci: nad primkou",
            marker=dict(color="red", size=10, symbol="x"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_pod_x,
            y=test_pod_y,
            mode="markers",
            name="Testovaci: pod primkou",
            marker=dict(color="blue", size=10, symbol="x"),
        )
    )

    fig.update_layout(
        title=f"Trenovany Perceptron<br>Train: {train_acc:.1f}%, Test: {test_acc:.1f}%",
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
