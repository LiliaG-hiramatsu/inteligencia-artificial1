import numpy as np
from collections import Counter
from knn import KNN
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# VALIDACIÓN CRUZADA K-FOLD DESDE CERO
# ─────────────────────────────────────────────

def kfold_split(X, y, n_folds=5, semilla=42):
    """
    Divide el dataset en n_folds partes estratificadas
    (igual proporción de cada clase en cada fold)
    """
    np.random.seed(semilla)
    clases = np.unique(y)
    indices_folds = [[] for _ in range(n_folds)]

    # Estratificación: dividir por clase para mantener proporción
    for clase in clases:
        idx_clase = np.where(y == clase)[0]
        np.random.shuffle(idx_clase)
        splits = np.array_split(idx_clase, n_folds)
        for i, split in enumerate(splits):
            indices_folds[i].extend(split)

    return indices_folds


def validacion_cruzada(X, y, k_vecinos=1, n_folds=5):
    """
    Ejecuta K-Fold cross validation y retorna métricas promedio
    """
    folds = kfold_split(X, y, n_folds=n_folds)
    clases = ["papa", "zanahoria", "choclo", "berenjena"]
    n_clases = len(clases)

    accuracies = []
    matriz_total = np.zeros((n_clases, n_clases), dtype=int)

    print(f"\nValidación Cruzada {n_folds}-Fold | K={k_vecinos}")
    print("-" * 45)

    for i in range(n_folds):
        # Fold i es el test, el resto es train
        idx_test  = np.array(folds[i])
        idx_train = np.concatenate([folds[j] for j in range(n_folds) if j != i])

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        # Entrenar y predecir
        knn = KNN(k=k_vecinos)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Accuracy del fold
        acc = np.sum(y_pred == y_test) / len(y_test)
        accuracies.append(acc)

        # Acumular matriz de confusión
        for real, pred in zip(y_test, y_pred):
            matriz_total[real][pred] += 1

        print(f"  Fold {i+1}: {acc*100:.1f}%  "
              f"(train={len(X_train)}, test={len(X_test)})")

    # Resultados finales
    acc_promedio = np.mean(accuracies)
    acc_std      = np.std(accuracies)

    print("-" * 45)
    print(f"  Accuracy promedio: {acc_promedio*100:.1f}% ± {acc_std*100:.1f}%")

    return acc_promedio, acc_std, matriz_total


def buscar_mejor_k_cv(X, y, k_max=10, n_folds=5):
    """
    Busca el mejor K usando validación cruzada
    """

    ks          = range(1, k_max + 1)
    medias      = []
    desviaciones = []

    print("\n" + "="*50)
    print("BÚSQUEDA DEL MEJOR K CON VALIDACIÓN CRUZADA")
    print("="*50)

    for k in ks:
        acc_mean, acc_std, _ = validacion_cruzada(X, y, k_vecinos=k, n_folds=n_folds)
        medias.append(acc_mean * 100)
        desviaciones.append(acc_std * 100)

    # Gráfico con barras de error
    plt.figure(figsize=(9, 5))
    plt.errorbar(ks, medias, yerr=desviaciones,
                 fmt='bo-', linewidth=2, markersize=8,
                 capsize=5, capthick=2, ecolor='lightblue')
    plt.xlabel("Valor de K")
    plt.ylabel("Accuracy promedio (%)")
    plt.title(f"Validación Cruzada {n_folds}-Fold – Accuracy vs K")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cv_accuracy_vs_k.png", dpi=150)
    plt.show()

    mejor_k   = list(ks)[np.argmax(medias)]
    mejor_acc = max(medias)
    mejor_std = desviaciones[np.argmax(medias)]

    print("\n" + "="*50)
    print(f"  Mejor K: {mejor_k}")
    print(f"  Accuracy: {mejor_acc:.1f}% ± {mejor_std:.1f}%")
    print("="*50)

    return mejor_k


def reporte_final(X, y, mejor_k, n_folds=5):
    """
    Genera el reporte final con el mejor K
    """

    clases = ["papa", "zanahoria", "choclo", "berenjena"]
    acc_mean, acc_std, matriz = validacion_cruzada(
        X, y, k_vecinos=mejor_k, n_folds=n_folds
    )

    # Métricas por clase
    print("\n" + "="*55)
    print(f"{'REPORTE FINAL – KNN con K=' + str(mejor_k):^55}")
    print("="*55)
    print(f"Accuracy global: {acc_mean*100:.1f}% ± {acc_std*100:.1f}%")
    print("-"*55)
    print(f"{'Clase':<12} {'Precisión':>10} {'Recall':>10} {'F1':>10}")
    print("-"*55)

    for i, clase in enumerate(clases):
        tp = matriz[i][i]
        fp = np.sum(matriz[:, i]) - tp
        fn = np.sum(matriz[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)
        print(f"{clase:<12} {precision*100:>9.1f}% "
              f"{recall*100:>9.1f}% {f1*100:>9.1f}%")

    print("="*55)

    # Matriz de confusión acumulada
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(clases)))
    ax.set_yticks(range(len(clases)))
    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión Acumulada – K={mejor_k} ({n_folds}-Fold CV)")

    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if matriz[i, j] > matriz.max() / 2 else "black"
            ax.text(j, i, str(matriz[i, j]),
                    ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("cv_matriz_confusion.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Cargar datos
    X = np.load("X_reducido.npy")
    y = np.load("y_labels.npy")

    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} componentes PCA")

    # Buscar mejor K con validación cruzada
    mejor_k = buscar_mejor_k_cv(X, y, k_max=10, n_folds=5)

    # Reporte final con el mejor K
    reporte_final(X, y, mejor_k, n_folds=5)
