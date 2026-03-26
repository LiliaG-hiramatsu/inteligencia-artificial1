# ─────────────────────────────────────────────
# PRUEBA 9-03-2026
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────
# 1. KNN DESDE CERO
# ─────────────────────────────────────────────

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Metodos privados
    def _distancia_euclidiana(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _predecir_uno(self, x):
        # Calcular distancia de x a todos los puntos de entrenamiento
        distancias = [self._distancia_euclidiana(x, x_train)
                      for x_train in self.X_train]

        # Obtener los k índices más cercanos
        indices_k = np.argsort(distancias)[:self.k]

        # Obtener las etiquetas de esos k vecinos
        etiquetas_k = [self.y_train[i] for i in indices_k]

        # Retornar la etiqueta más frecuente (voto mayoritario)
        return Counter(etiquetas_k).most_common(1)[0][0]

    def predict(self, X_test):
        return np.array([self._predecir_uno(x) for x in X_test])


# ─────────────────────────────────────────────
# 2. DIVISIÓN TRAIN / TEST DESDE CERO
# ─────────────────────────────────────────────

def train_test_split_propio(X, y, test_size=0.2, semilla=42):
    np.random.seed(semilla)
    n = len(X)
    indices = np.random.permutation(n)
    corte = int(n * (1 - test_size))
    idx_train = indices[:corte]
    idx_test  = indices[corte:]
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test]


# ─────────────────────────────────────────────
# 3. MÉTRICAS DESDE CERO
# ─────────────────────────────────────────────

def calcular_metricas(y_real, y_pred, clases):
    n_clases = len(clases)
    
    # Matriz de confusión
    matriz = np.zeros((n_clases, n_clases), dtype=int)
    for real, pred in zip(y_real, y_pred):
        matriz[real][pred] += 1

    # Precisión global
    accuracy = np.sum(y_real == y_pred) / len(y_real)

    # Precisión, recall y F1 por clase
    print("\n" + "="*55)
    print(f"{'RESULTADOS KNN':^55}")
    print("="*55)
    print(f"Accuracy global: {accuracy*100:.1f}%")
    print(f"Muestras de prueba: {len(y_real)}")
    print("-"*55)
    print(f"{'Clase':<12} {'Precisión':>10} {'Recall':>10} {'F1':>10}")
    print("-"*55)

    for i, clase in enumerate(clases):
        tp = matriz[i][i]
        fp = np.sum(matriz[:, i]) - tp
        fn = np.sum(matriz[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0)

        print(f"{clase:<12} {precision*100:>9.1f}% {recall*100:>9.1f}% {f1*100:>9.1f}%")

    print("="*55)
    return matriz, accuracy


def graficar_confusion(matriz, clases):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, cmap="Blues")

    ax.set_xticks(range(len(clases)))
    ax.set_yticks(range(len(clases)))
    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión – KNN")

    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if matriz[i, j] > matriz.max() / 2 else "black"
            ax.text(j, i, str(matriz[i, j]),
                    ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=150)
    plt.show()


def graficar_accuracy_vs_k(X_train, y_train, X_test, y_test, k_max=10):
    ks = range(1, k_max + 1)
    accuracies = []

    for k in ks:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = np.sum(y_pred == y_test) / len(y_test)
        accuracies.append(acc * 100)

    plt.figure(figsize=(7, 4))
    plt.plot(ks, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Valor de K")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs K – KNN")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_vs_k.png", dpi=150)
    plt.show()

    mejor_k = ks[np.argmax(accuracies)]
    print(f"\nMejor K: {mejor_k} ({max(accuracies):.1f}% accuracy)")
    return mejor_k


# ─────────────────────────────────────────────
# 4. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Cargar datos guardados por el script anterior
    X = np.load("X_reducido.npy")
    y = np.load("y_labels.npy")
    clases = ["papa", "zanahoria", "choclo", "berenjena"]
    print("🥔 🥕 🌽 🍆")
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} componentes PCA")

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split_propio(
        X, y, test_size=0.2, semilla=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Encontrar el mejor K
    print("\nBuscando mejor K...")
    mejor_k = graficar_accuracy_vs_k(X_train, y_train, X_test, y_test, k_max=10)

    # Entrenar con el mejor K
    knn = KNN(k=mejor_k)
    knn.fit(X_train, y_train)

    # Predecir y evaluar
    y_pred = knn.predict(X_test)
    matriz, accuracy = calcular_metricas(y_test, y_pred, clases)
    graficar_confusion(matriz, clases)

    # Guardar modelo
    np.save("knn_X_train.npy", X_train)
    np.save("knn_y_train.npy", y_train)
    print(f"\n✅Modelo Knn guardado. Listo para predecir nuevos audios.")