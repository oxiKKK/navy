import matplotlib.pyplot as plt
import numpy as np

NUM_CHANGES = 6


# Nahrazeni 0 za -1
def minusone_for_zero(vzor):
    return np.where(np.array(vzor) == 0, -1, 1)


def signum(hodnota, predchozi_stav):
    s = np.sign(hodnota)
    # Pokud je signum 0, stav se nemění
    return np.where(s == 0, predchozi_stav, s)


def train(vzory):
    velikost = len(vzory[0])
    W = np.zeros((velikost, velikost))

    # Pro kazdy vzor vytvorime vahovou matici
    # a pridame ji k celkove vahove matici W
    for vzor in vzory:
        X = vzor.reshape(-1, 1)  # sloupcovy vektor
        W += X @ X.T

    # diagonala na 0
    np.fill_diagonal(W, 0)
    return W


def synchronni_recovery(W, vzor, kroky=5):
    # vsechny neurony se auktualizuji najednou
    X = vzor.copy()
    for _ in range(kroky):
        X = signum(W @ X, X)
    return X


def asynchronni_recovery(W, vzor, pruchody=5):
    # neurony se aktualizuji postupne
    X = vzor.copy()
    velikost = len(X)
    for _ in range(pruchody):
        for i in range(velikost):
            X[i] = signum(W[i] @ X, X[i])
    return X


def pridej_sum(vzor):
    # nahodne zmenim znaky v vzoru
    result = vzor.copy()
    indexy = np.random.choice(len(vzor), NUM_CHANGES, replace=False)
    result[indexy] *= -1
    return result


def render(nazvy, vzory, W):
    _, axes = plt.subplots(3, 4, figsize=(10, 7))
    titulky = ["Puvodni", "Zniceny", "Sync", "Async"]

    for radek, (nazev, orig) in enumerate(zip(nazvy, vzory)):
        poskozeny = pridej_sum(orig)

        sync_obnoveny = synchronni_recovery(W, poskozeny)
        async_obnoveny = asynchronni_recovery(W, poskozeny)

        obrazky = [orig, poskozeny, sync_obnoveny, async_obnoveny]

        for sloupec, data_obrazku in enumerate(obrazky):
            ax = axes[radek, sloupec]
            ax.imshow(data_obrazku.reshape(5, 5), cmap="binary", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if radek == 0:
                ax.set_title(titulky[sloupec])
            if sloupec == 0:
                ax.set_ylabel(nazev, rotation=0, labelpad=15, fontsize=14)

    plt.tight_layout()
    plt.savefig("images/hopfield_recovery.png", dpi=150)
    print("img saved")


def main():
    # vzory H, T, X
    nazvy = ["H", "T", "X"]
    H = [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]
    T = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    X = [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]

    # prevedeni 0 na -1
    vzory = [minusone_for_zero(p) for p in [H, T, X]]

    # trenovani site
    W = train(vzory)

    # vykresleni
    render(nazvy, vzory, W)


if __name__ == "__main__":
    main()
