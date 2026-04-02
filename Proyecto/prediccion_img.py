import numpy as np
from PIL import Image
from kmeans import extraer_caracteristicas_imagen, KMeans_propio
from entrenamiento import PCA_propio
import matplotlib.pyplot as plt
import os

# ── Cargar modelo ──
clases      = ["papa", "zanahoria", "choclo", "berenjena"]
centroides  = np.load("kmeans_centroides.npy")
media_norm  = np.load("img_media_norm.npy")
std_norm    = np.load("img_std_norm.npy")
mapa_arr    = np.load("img_mapa_clusters.npy")
pca_comp    = np.load("img_pca_componentes.npy")
pca_media   = np.load("img_pca_media.npy")
mapa        = {int(k): int(v) for k, v in mapa_arr}

kmeans = KMeans_propio(k=4)
kmeans.centroides = centroides

# ── Función de predicción para una imagen ──
def predecir_imagen(ruta):
    feat = extraer_caracteristicas_imagen(ruta)
    feat_norm = (feat - media_norm) / (std_norm + 1e-8)
    feat_pca  = (feat_norm - pca_media) @ pca_comp
    cluster   = kmeans.predict(feat_pca.reshape(1, -1))[0]
    clase_idx = mapa[cluster]
    return clases[clase_idx]

# ── Probar con imágenes de test ──
carpeta_test = "test_img/"

print(f"Features esperadas por el modelo: {media_norm.shape[0]}")
feat_prueba = extraer_caracteristicas_imagen(os.path.join(carpeta_test, os.listdir(carpeta_test)[0]))
print(f"Features que genera extraer_caracteristicas_imagen: {feat_prueba.shape[0]}")

for archivo in os.listdir(carpeta_test):
    if not archivo.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    ruta = os.path.join(carpeta_test, archivo)
    prediccion = predecir_imagen(ruta)
    print(f"{archivo:40s} → {prediccion}")