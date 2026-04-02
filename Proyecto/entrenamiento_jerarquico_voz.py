import numpy as np
import matplotlib.pyplot as plt

from entrenamiento import normalizar, PCA_propio

# ─────────────────────────────────────────────
# ENTRENAMIENTO DEL KNN ESPECIALIZADO
# Clases: zanahoria (1), choclo (2), berenjena (3)
# ─────────────────────────────────────────────

# Cargar datos crudos (antes de normalizar y PCA)
X_raw = np.load("X_features_raw.npy")
y     = np.load("y_labels.npy")

# Filtrar solo las 3 clases que se confunden
clases_sub = [1, 2, 3]  # zanahoria, choclo, berenjena
mask = np.isin(y, clases_sub)
X_sub = X_raw[mask]
y_sub = y[mask]

print(f"Sub-dataset: {X_sub.shape[0]} muestras, clases: {np.unique(y_sub)}")

# Normalizar con estadísticas PROPIAS del sub-dataset
X_sub_norm, media_sub, std_sub = normalizar(X_sub)

"""
# Ver varianza acumulada para elegir n_componentes
pca_full = PCA_propio(n_componentes=X_sub_norm.shape[1])
pca_full.fit(X_sub_norm)
acumulada = np.cumsum(pca_full.varianza_explicada)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(acumulada)+1), acumulada, 'bo-')
plt.axhline(95, color='r', linestyle='--', label='95% varianza')
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada (%)")
plt.title("Scree Plot – Sub-clasificador")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

# PCA propio del sub-dataset
pca_sub = PCA_propio(n_componentes=26)
X_sub_reducido = pca_sub.fit_transform(X_sub_norm)

print(f"Dimensión reducida sub-KNN: {X_sub_reducido.shape}")

# Guardar todo
np.save("sub_X_train.npy",        X_sub_reducido)
np.save("sub_y_train.npy",        y_sub)
np.save("sub_norm_media.npy",     media_sub)
np.save("sub_norm_std.npy",       std_sub)
np.save("sub_pca_componentes.npy", pca_sub.componentes)
np.save("sub_pca_media.npy",       pca_sub.media)

print("Modelo especializado guardado.")
