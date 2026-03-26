import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
from kmeans import KMeans_propio, extraer_caracteristicas_imagen

# ── Recrear el objeto kmeans con los centroides guardados ──
centroides = np.load("kmeans_centroides.npy")

kmeans = KMeans_propio(k=4)
kmeans.centroides = centroides   # cargar centroides directamente

# ── Cargar normalización ──
media_norm = np.load("img_media_norm.npy")
std_norm   = np.load("img_std_norm.npy")

# ── Cargar mapa de clusters ──
mapa_arr = np.load("img_mapa_clusters.npy")
mapa     = {int(k): int(v) for k, v in mapa_arr}

clases = ["papa", "zanahoria", "choclo", "berenjena"]

def visualizar_clusters(carpeta_raiz, kmeans, media_norm, std_norm, clases, mapa):
    """
    Muestra una grilla de imágenes por cluster para ver qué metió K-means en cada grupo
    """
    formatos = [".jpg", ".jpeg", ".png"]
    rutas, y_real = [], []

    for idx, verdura in enumerate(clases):
        carpeta = os.path.join(carpeta_raiz, verdura)
        for archivo in os.listdir(carpeta):
            if any(archivo.lower().endswith(e) for e in formatos):
                rutas.append(os.path.join(carpeta, archivo))
                y_real.append(idx)

    # Extraer features y predecir clusters
    X = []
    for ruta in rutas:
        X.append(extraer_caracteristicas_imagen(ruta))
    X = np.array(X)
    X_norm = (X - media_norm) / (std_norm + 1e-8)
    etiquetas = kmeans.predict(X_norm)

    # Graficar 5 imágenes de cada cluster
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for k in range(4):
        indices = np.where(etiquetas == k)[0]
        verdura_asignada = clases[mapa[k]]
        for j in range(5):
            ax = axes[k][j]
            if j < len(indices):
                img = mpimg.imread(rutas[indices[j]])
                ax.imshow(img)
                verdura_real = clases[y_real[indices[j]]]
                color = "green" if verdura_real == verdura_asignada else "red"
                ax.set_title(verdura_real, color=color, fontsize=8)
            ax.axis("off")
        axes[k][0].set_ylabel(f"Cluster {k}\n→{verdura_asignada}",
                              fontsize=9, rotation=0, labelpad=60)

    plt.suptitle("Contenido de cada cluster (verde=correcto, rojo=error)", fontsize=12)
    plt.tight_layout()
    plt.savefig("clusters_visualizacion.png", dpi=120)
    plt.show()


# ── Llamar después de entrenar ──
media_norm = np.load("img_media_norm.npy")
std_norm   = np.load("img_std_norm.npy")
mapa_arr   = np.load("img_mapa_clusters.npy")
mapa       = {int(k): int(v) for k, v in mapa_arr}
clases     = ["papa", "zanahoria", "choclo", "berenjena"]

visualizar_clusters("dataset_img/", kmeans, media_norm, std_norm, clases, mapa)