# ─────────────────────────────────────────────
# PRUEBA 9-03-2026
# ─────────────────────────────────────────────

import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# 1. EXTRACCIÓN DE CARACTERÍSTICAS
# ─────────────────────────────────────────────

def extraer_caracteristicas(ruta_audio, n_mfcc=13):
    y, sr = librosa.load(ruta_audio, sr=None)

    # Eliminar silencios al inicio y fin
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalizar amplitud
    y = librosa.util.normalize(y)
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std  = np.std(mfccs, axis=1)

    # Delta MFCCs (velocidad de cambio)
    delta = librosa.feature.delta(mfccs)
    delta_mean = np.mean(delta, axis=1)
    delta_std  = np.std(delta, axis=1)

    # Características complementarias
    zcr     = np.mean(librosa.feature.zero_crossing_rate(y))
    rms     = np.mean(librosa.feature.rms(y=y))
    centroid= np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    return np.concatenate([
        mfccs_mean, mfccs_std,
        delta_mean, delta_std,
        [zcr, rms, centroid, rolloff]
    ])


def cargar_dataset(carpeta_raiz):
    """
    Estructura esperada:
        dataset_voz/
            papa/       *.wav
            zanahoria/  *.wav
            choclo/     *.wav
            berenjena/  *.wav
    """
    clases = ["papa", "zanahoria", "choclo", "berenjena"]
    X, y = [], []

    for idx, verdura in enumerate(clases):
        carpeta = os.path.join(carpeta_raiz, verdura)
        for archivo in os.listdir(carpeta):
            if archivo.endswith(".wav"):
                ruta = os.path.join(carpeta, archivo)
                try:
                    feat = extraer_caracteristicas(ruta)
                    X.append(feat)
                    y.append(idx)        # etiqueta numérica
                    print(f"✓ {verdura}/{archivo}")
                except Exception as e:
                    print(f"✗ {archivo}: {e}")

    return np.array(X), np.array(y), clases


# ─────────────────────────────────────────────
# 2. PCA IMPLEMENTADO DESDE CERO
# ─────────────────────────────────────────────

class PCA_propio:
    def __init__(self, n_componentes=2):
        self.n_componentes = n_componentes
        self.componentes = None
        self.media = None
        self.varianza_explicada = None

    def fit(self, X):
        self.media = np.mean(X, axis=0)
        X_centrado = X - self.media

        # Matriz de covarianza
        cov = np.cov(X_centrado.T)

        # Autovalores y autovectores
        autovalores, autovectores = np.linalg.eigh(cov)

        # Ordenar de mayor a menor varianza
        orden = np.argsort(autovalores)[::-1]
        autovalores  = autovalores[orden]
        autovectores = autovectores[:, orden]

        # Guardar los primeros n componentes
        self.componentes = autovectores[:, :self.n_componentes]

        # Varianza explicada (%)
        total = np.sum(autovalores)
        self.varianza_explicada = (autovalores[:self.n_componentes] / total) * 100

    def transform(self, X):
        X_centrado = X - self.media
        return X_centrado @ self.componentes

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ─────────────────────────────────────────────
# 3. NORMALIZACIÓN (recomendada antes del PCA)
# ─────────────────────────────────────────────

def normalizar(X):
    media = np.mean(X, axis=0)
    std   = np.std(X, axis=0) + 1e-8   # evitar división por cero
    return (X - media) / std, media, std


# ─────────────────────────────────────────────
# 4. VISUALIZACIÓN DE CLASES
# ─────────────────────────────────────────────

COLORES = ["royalblue", "tomato", "gold", "mediumseagreen"]

def graficar_2D(X_pca, y, clases, pca):
    plt.figure(figsize=(8, 6))
    for idx, nombre in enumerate(clases):
        mask = y == idx
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=COLORES[idx], label=nombre,
                    edgecolors='k', s=80, alpha=0.85)

    plt.xlabel(f"PC1 ({pca.varianza_explicada[0]:.1f}% var.)")
    plt.ylabel(f"PC2 ({pca.varianza_explicada[1]:.1f}% var.)")
    plt.title("PCA 2D – Separación de clases (VOZ)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pca_2d_voz.png", dpi=150)
    plt.show()


def graficar_3D(X_pca, y, clases, pca):
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    for idx, nombre in enumerate(clases):
        mask = y == idx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   color=COLORES[idx], label=nombre,
                   edgecolors='k', s=60, alpha=0.85)

    ax.set_xlabel(f"PC1 ({pca.varianza_explicada[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.varianza_explicada[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.varianza_explicada[2]:.1f}%)")
    ax.set_title("PCA 3D – Separación de clases (VOZ)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pca_3d_voz.png", dpi=150)
    plt.show()


def graficar_varianza_acumulada(pca_full):
    """Scree plot: cuántas componentes capturan el 95% de varianza"""
    acumulada = np.cumsum(pca_full.varianza_explicada)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(acumulada)+1), acumulada, 'bo-')
    plt.axhline(95, color='r', linestyle='--', label='95% varianza')
    plt.xlabel("Número de componentes")
    plt.ylabel("Varianza explicada acumulada (%)")
    plt.title("Scree Plot – Selección de componentes PCA")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("scree_plot.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 5. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # --- Cargar y extraer ---
    X, y, clases = cargar_dataset("dataset_audio/")
    print(f"\nDataset: {X.shape[0]} muestras, {X.shape[1]} características")

    # --- Guardar features RAW (antes de normalizar y PCA) ---
    np.save("X_features_raw.npy", X)
    
    # --- Normalizar ---
    X_norm, media_norm, std_norm = normalizar(X)

    # --- PCA completo para scree plot ---
    n_total = X_norm.shape[1]
    pca_full = PCA_propio(n_componentes=n_total)
    pca_full.fit(X_norm)
    graficar_varianza_acumulada(pca_full)   # decidir cuántas componentes usar

    # --- PCA 2D para visualización ---
    pca_2d = PCA_propio(n_componentes=2)
    X_2d   = pca_2d.fit_transform(X_norm)
    graficar_2D(X_2d, y, clases, pca_2d)

    # --- PCA 3D para visualización ---
    pca_3d = PCA_propio(n_componentes=3)
    X_3d   = pca_3d.fit_transform(X_norm)
    graficar_3D(X_3d, y, clases, pca_3d)

    # PCA final (reducción real para el KNN)
    # Elegir n según el scree plot (típicamente donde se llega a ~95%)
    pca_knn = PCA_propio(n_componentes=20)
    X_reducido = pca_knn.fit_transform(X_norm)
    print(f"Dimensión reducida para KNN: {X_reducido.shape}")
    
    # Guardar también la media del PCA
    np.save("pca_media.npy", pca_knn.media)

    # Guardar todo lo necesario para el KNN
    np.save("X_reducido.npy", X_reducido)
    np.save("y_labels.npy", y)
    np.save("pca_componentes.npy", pca_knn.componentes)
    np.save("norm_media.npy", media_norm)
    np.save("norm_std.npy", std_norm)
    print("\nDatos guardados. Listos para entrenar KNN.")