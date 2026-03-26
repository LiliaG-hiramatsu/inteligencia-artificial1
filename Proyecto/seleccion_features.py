import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from entrenamiento import PCA_propio

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def seleccionar_features(X, y, n_features=15):
    """
    Selecciona las n_features más discriminativas usando
    el criterio de Fisher: varianza entre clases / varianza dentro de clases
    Cuanto mayor el score, mejor separa esa feature las clases.
    """
    clases     = np.unique(y)
    n_features_total = X.shape[1]
    scores     = np.zeros(n_features_total)
    media_global = np.mean(X, axis=0)

    for f in range(n_features_total):
        # Varianza ENTRE clases (qué tan separados están los centros)
        var_entre = 0
        for c in clases:
            X_clase   = X[y == c, f]
            n_c       = len(X_clase)
            media_c   = np.mean(X_clase)
            var_entre += n_c * (media_c - media_global[f]) ** 2

        # Varianza DENTRO de clases (qué tan dispersa está cada clase)
        var_dentro = 0
        for c in clases:
            X_clase    = X[y == c, f]
            media_c    = np.mean(X_clase)
            var_dentro += np.sum((X_clase - media_c) ** 2)

        # Score de Fisher
        scores[f] = var_entre / (var_dentro + 1e-8)

    # Índices de las mejores features ordenadas de mayor a menor
    indices_top = np.argsort(scores)[::-1][:n_features]

    print(f"\nTop {n_features} features más discriminativas:")
    print(f"Índices: {indices_top}")
    print(f"Scores:  {scores[indices_top].round(3)}")

    return indices_top, scores


def graficar_scores(scores, n_top=20):
    """Muestra qué features son más discriminativas"""
    indices_ord = np.argsort(scores)[::-1]
    plt.figure(figsize=(12, 4))
    colores = ['tomato' if i < n_top else 'steelblue'
               for i in range(len(scores))]
    colores_ord = [colores[i] for i in indices_ord]
    plt.bar(range(len(scores)), scores[indices_ord], color=colores_ord)
    plt.axvline(x=n_top - 0.5, color='red', linestyle='--',
                label=f'Top {n_top} seleccionadas')
    plt.xlabel("Feature (ordenada por score)")
    plt.ylabel("Score de Fisher")
    plt.title("Discriminabilidad de cada feature (Criterio de Fisher)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fisher_scores.png", dpi=150)
    plt.show()


COLORES = ["royalblue", "tomato", "gold", "mediumseagreen"]

def graficar_pca_2d(X_pca, y, clases, pca, titulo="PCA 2D"):
    plt.figure(figsize=(8, 6))
    for idx, nombre in enumerate(clases):
        mask = y == idx
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=COLORES[idx], label=nombre,
                    edgecolors='k', s=100, alpha=0.85)

        # Centroide de cada clase
        cx = np.mean(X_pca[mask, 0])
        cy = np.mean(X_pca[mask, 1])
        plt.scatter(cx, cy, color=COLORES[idx],
                    marker='*', s=300, edgecolors='k', zorder=5)

    plt.xlabel(f"PC1 ({pca.varianza_explicada[0]:.1f}% var.)")
    plt.ylabel(f"PC2 ({pca.varianza_explicada[1]:.1f}% var.)")
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pca_2d_fisher.png", dpi=150)
    plt.show()


def graficar_pca_3d(X_pca, y, clases, pca, titulo="PCA 3D"):
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')
    for idx, nombre in enumerate(clases):
        mask = y == idx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   color=COLORES[idx], label=nombre,
                   edgecolors='k', s=60, alpha=0.85)

        # Centroide
        cx = np.mean(X_pca[mask, 0])
        cy = np.mean(X_pca[mask, 1])
        cz = np.mean(X_pca[mask, 2])
        ax.scatter(cx, cy, cz, color=COLORES[idx],
                   marker='*', s=300, edgecolors='k', zorder=5)

    ax.set_xlabel(f"PC1 ({pca.varianza_explicada[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.varianza_explicada[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.varianza_explicada[2]:.1f}%)")
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    plt.savefig("pca_3d_fisher.png", dpi=150)
    plt.show()


def elipse_confianza(ax, x, y, color, n_std=1.5):
    """Dibuja una elipse que encierra ~86% de los puntos de la clase"""
    cov  = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    orden = np.argsort(vals)[::-1]
    vals  = vals[orden]
    vecs  = vecs[:, orden]

    angulo = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ancho  = 2 * n_std * np.sqrt(vals[0])
    alto   = 2 * n_std * np.sqrt(vals[1])

    elipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=ancho, height=alto,
        angle=angulo,
        facecolor=color, alpha=0.15,
        edgecolor=color, linewidth=2
    )
    ax.add_patch(elipse)


def graficar_pca_2d_con_elipses(X_pca, y, clases, pca):
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, nombre in enumerate(clases):
        mask = y == idx
        x_c  = X_pca[mask, 0]
        y_c  = X_pca[mask, 1]

        # Puntos
        ax.scatter(x_c, y_c, color=COLORES[idx], label=nombre,
                   edgecolors='k', s=100, alpha=0.85, zorder=3)

        # Centroide
        ax.scatter(np.mean(x_c), np.mean(y_c),
                   color=COLORES[idx], marker='*',
                   s=400, edgecolors='k', zorder=5)

        # Elipse de confianza
        if len(x_c) > 2:
            elipse_confianza(ax, x_c, y_c, COLORES[idx])

    ax.set_xlabel(f"PC1 ({pca.varianza_explicada[0]:.1f}% var.)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.varianza_explicada[1]:.1f}% var.)", fontsize=12)
    ax.set_title("PCA 2D – Separación de clases con elipses de confianza", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pca_2d_elipses.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Cargar features originales (sin reducir)
    X_raw = np.load("X_features_raw.npy")   # ← features sin PCA
    y     = np.load("y_labels.npy")
    clases = ["papa", "zanahoria", "choclo", "berenjena"]

    # Normalizar
    media = np.mean(X_raw, axis=0)
    std   = np.std(X_raw,  axis=0) + 1e-8
    X_norm = (X_raw - media) / std

    # Seleccionar mejores features con Fisher
    indices_top, scores = seleccionar_features(X_norm, y, n_features=15)
    # graficar_scores(scores)

    # Quedarse solo con esas features
    X_seleccionado = X_norm[:, indices_top]

    # Aplicar PCA sobre las features seleccionadas
    pca_2d = PCA_propio(n_componentes=2)
    X_2d   = pca_2d.fit_transform(X_seleccionado)
    # graficar_pca_2d(X_2d, y, clases, pca_2d,
    #                titulo="PCA 2D – Features seleccionadas (Fisher)")

    pca_3d = PCA_propio(n_componentes=3)
    X_3d   = pca_3d.fit_transform(X_seleccionado)
    # graficar_pca_3d(X_3d, y, clases, pca_3d,
    #                 titulo="PCA 3D – Features seleccionadas (Fisher)")
    
    # ── Llamar en lugar de graficar_pca_2d ──
    graficar_pca_2d_con_elipses(X_2d, y, clases, pca_2d)