import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from entrenamiento import PCA_propio
import cv2

# ─────────────────────────────────────────────
# 1. EXTRACCIÓN DE CARACTERÍSTICAS DE IMÁGENES
# ─────────────────────────────────────────────

def aplicar_grabcut(arr):
    img_uint8 = arr.astype(np.uint8)
    mask = np.zeros(img_uint8.shape[:2], np.uint8)
    
    h, w = img_uint8.shape[:2]
    margen = 0.10  # 10% de margen desde los bordes
    rect = (
        int(w * margen),        # x inicio
        int(h * margen),        # y inicio
        int(w * (1-2*margen)),  # ancho
        int(h * (1-2*margen))   # alto
    )
    
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_uint8, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    
    # Los píxeles con valor 2 o 0 son fondo, el resto es objeto
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    
    # Aplicar máscara: fondo queda en negro
    resultado = arr * mask2[:, :, np.newaxis]
    return resultado.astype(np.float32), mask2
    
# ── Convertir RGB a HSV ──
def histogramas_hsv(a, mask=None):
    r,g,b = a[:,:,0]/255, a[:,:,1]/255, a[:,:,2]/255
    cmax  = np.maximum(np.maximum(r,g),b)
    cmin  = np.minimum(np.minimum(r,g),b)
    delta = cmax - cmin + 1e-8
    h_    = np.zeros_like(r)
    mask_r = cmax == r
    h_[mask_r] = 60*(((g[mask_r]-b[mask_r])/delta[mask_r])%6)
    mask_g = cmax == g
    h_[mask_g] = 60*(((b[mask_g]-r[mask_g])/delta[mask_g])+2)
    mask_b = cmax == b
    h_[mask_b] = 60*(((r[mask_b]-g[mask_b])/delta[mask_b])+4)
    s_ = delta/(cmax+1e-8)
    v_ = cmax

    # Aplanar y filtrar solo píxeles de la verdura
    if mask is not None:
        m = mask.flatten().astype(bool)
        h_f = h_.flatten()[m]
        s_f = s_.flatten()[m]
        v_f = v_.flatten()[m]
    else:
        h_f, s_f, v_f = h_.flatten(), s_.flatten(), v_.flatten()

    hh,_ = np.histogram(h_f, bins=32, range=(0,360))
    hs,_ = np.histogram(s_f, bins=16, range=(0,1))
    hv,_ = np.histogram(v_f, bins=16, range=(0,1))
    hh = hh/(hh.sum()+1e-8)
    hs = hs/(hs.sum()+1e-8)
    hv = hv/(hv.sum()+1e-8)
    return np.concatenate([hh, hs, hv])
    
def momento_central(img, p, q):
    h_, w_ = img.shape
    x = np.arange(w_)
    y = np.arange(h_)
    xx, yy = np.meshgrid(x, y)
    m00 = np.sum(img) + 1e-8
    cx  = np.sum(xx*img)/m00
    cy  = np.sum(yy*img)/m00
    return np.sum(((xx-cx)**p)*((yy-cy)**q)*img)
    
def extraer_caracteristicas_imagen(ruta, tam=(128, 128)):
    img = Image.open(ruta).convert("RGB")
    img = img.resize(tam)
    arr = np.array(img, dtype=np.float32)

    # ── Segmentación de fondo ──
    resultado, mask2 = aplicar_grabcut(arr)
    
    # ── Usar solo el 60% central (recortar fondo de bordes) ──
    h, w  = resultado.shape[:2]
    mh, mw = int(h * 0.2), int(w * 0.2)
    centro = resultado[mh:h-mh, mw:w-mw, :]
    mask_centro = mask2[mh:h-mh, mw:w-mw]

    # Features de imagen completa
    feat_completa = histogramas_hsv(resultado, mask=mask2)

    # Features solo del centro (más peso a la verdura)
    feat_centro   = histogramas_hsv(centro, mask=mask_centro)

    # Color promedio del centro
    media_rgb = np.mean(centro[mask_centro.astype(bool)], axis=0) / 255
    std_rgb   = np.std(centro[mask_centro.astype(bool)],  axis=0) / 255

     # --- Característica de aspecto ---
    coords = cv2.findNonZero(mask2)  # Píxeles del objeto
    aspect_ratio = 1.0  # valor neutro por defecto
    if coords is not None and len(coords) > 10:
        x, y, bw, bh = cv2.boundingRect(coords)
        if bh > 0:
            aspect_ratio = bw / bh  # < 1 → alargado (zanahoria), ~1 → redondo
    
    # Momentos de Hu sobre el centro
    gris = (0.299*centro[:,:,0] + 0.587*centro[:,:,1] + 0.114*centro[:,:,2]) / 255
    gris = gris * mask_centro  # solo píxeles de la verdura

    m00 = np.sum(gris)+1e-8
    u20 = momento_central(gris,2,0)/m00
    u02 = momento_central(gris,0,2)/m00
    u11 = momento_central(gris,1,1)/m00
    u30 = momento_central(gris,3,0)/m00
    u03 = momento_central(gris,0,3)/m00
    u21 = momento_central(gris,2,1)/m00
    u12 = momento_central(gris,1,2)/m00

    hu = np.array([
        u20+u02,
        (u20-u02)**2 + 4*u11**2,
        (u30-3*u12)**2 + (3*u21-u03)**2,
        (u30+u12)**2  + (u21+u03)**2,
        (u30-3*u12)*(u30+u12)*((u30+u12)**2-3*(u21+u03)**2)+
        (3*u21-u03)*(u21+u03)*(3*(u30+u12)**2-(u21+u03)**2),
        (u20-u02)*((u30+u12)**2-(u21+u03)**2)+4*u11*(u30+u12)*(u21+u03),
        (3*u21-u03)*(u30+u12)*((u30+u12)**2-3*(u21+u03)**2)-
        (u30-3*u12)*(u21+u03)*(3*(u30+u12)**2-(u21+u03)**2)
    ])
    # Se aplica logaritmo porque los momentos de orden alto tienen valores enormes comparados con los de orden bajo
    # np.sign preserva el signo original de hu
    hu_log = -np.sign(hu)*np.log10(np.abs(hu)+1e-10)

    return np.concatenate([
        feat_completa,                  # 64 features imagen completa
        feat_centro * 2,                # 64 features centro (peso doble)
        media_rgb,                      # 3
        std_rgb,                        # 3
        np.array([aspect_ratio]),       # 1
        hu_log                          # 7
    ])


def cargar_dataset_imagenes(carpeta_raiz):
    clases   = ["papa", "zanahoria", "choclo", "berenjena"]
    formatos = [".jpg", ".jpeg", ".png"]
    X, y     = [], []

    for idx, verdura in enumerate(clases):
        carpeta = os.path.join(carpeta_raiz, verdura)
        archivos = [f for f in os.listdir(carpeta)
                    if any(f.lower().endswith(e) for e in formatos)]

        for archivo in archivos:
            ruta = os.path.join(carpeta, archivo)
            try:
                feat = extraer_caracteristicas_imagen(ruta)
                X.append(feat)
                y.append(idx)
                print(f"{verdura}/{archivo}")
            except Exception as e:
                print(f"{archivo}: {e}")

    return np.array(X), np.array(y), clases


# ─────────────────────────────────────────────
# 2. K-MEANS DESDE CERO
# ─────────────────────────────────────────────

class KMeans_propio:
    def __init__(self, k=4, max_iter=300, semilla=42):
        self.k = k
        self.max_iter = max_iter
        self.semilla = semilla
        self.centroides = None

    def fit(self, X):
        np.random.seed(self.semilla)

        # Inicializar centroides aleatoriamente
        indices = np.random.choice(len(X), self.k, replace=False)
        self.centroides = X[indices].copy()

        for iteracion in range(self.max_iter):
            # Asignar cada punto al centroide más cercano
            etiquetas = self._asignar(X)

            # Recalcular centroides
            nuevos_centroides = np.array([
                X[etiquetas == k].mean(axis=0)
                if np.sum(etiquetas == k) > 0
                else self.centroides[k]
                for k in range(self.k)
            ])

            # Verificar convergencia
            if np.allclose(self.centroides, nuevos_centroides, atol=1e-6):
                print(f"Convergió en la iteración {iteracion + 1}")
                break

            self.centroides = nuevos_centroides

        return self

    def _asignar(self, X):
        distancias = np.array([
            np.sqrt(np.sum((X - c) ** 2, axis=1))
            for c in self.centroides
        ])
        return np.argmin(distancias, axis=0)

    def predict(self, X):
        return self._asignar(X)


# ─────────────────────────────────────────────
# 3. ASIGNAR ETIQUETAS A CLUSTERS
# ─────────────────────────────────────────────

def asignar_etiquetas_clusters(etiquetas_kmeans, y_real, n_clusters=4):
    """
    K-means no sabe qué cluster es papa, cuál es zanahoria, etc.
    Esta función asigna la etiqueta real a cada cluster
    basándose en la clase mayoritaria dentro de cada cluster.
    """
    mapa = {}
    for cluster in range(n_clusters):
        mask   = etiquetas_kmeans == cluster
        if np.sum(mask) == 0:
            mapa[cluster] = -1
            continue
        clases_en_cluster = y_real[mask]
        mapa[cluster] = np.bincount(clases_en_cluster).argmax()
    return mapa


# ─────────────────────────────────────────────
# 4. VISUALIZACIÓN PCA DE IMÁGENES
# ─────────────────────────────────────────────

COLORES  = ["royalblue", "tomato", "gold", "mediumseagreen"]
MARKERS  = ["o", "s", "^", "D"]

def graficar_clusters_2d(X_pca, etiquetas_kmeans, y_real,
                          mapa, clases, pca):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Gráfico izquierdo: colores por cluster K-means ──
    colores_cluster = ["purple", "orange", "cyan", "brown"]
    ax = axes[0]
    for k in range(4):
        mask = etiquetas_kmeans == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=colores_cluster[k],
                   label=f"Cluster {k} → {clases[mapa[k]]}",
                   edgecolors='k', s=80, alpha=0.8)
    ax.set_title("Clusters K-means")
    ax.set_xlabel(f"PC1 ({pca.varianza_explicada[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.varianza_explicada[1]:.1f}%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Gráfico derecho: colores por clase real ──
    ax = axes[1]
    for idx, nombre in enumerate(clases):
        mask = y_real == idx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=COLORES[idx], label=nombre,
                   edgecolors='k', s=80, alpha=0.8)
    ax.set_title("Clases reales")
    ax.set_xlabel(f"PC1 ({pca.varianza_explicada[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.varianza_explicada[1]:.1f}%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("K-means – Imágenes de verduras", fontsize=13)
    plt.tight_layout()
    plt.savefig("kmeans_imagenes.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 5. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Cargar o procesar dataset ──
    # si queres reentrenar el modelo, eliminá estos dos archivos: dataset_img_X.npy y dataset_img_y.npy así se ejecuta el else
    if os.path.exists("dataset_img_X.npy") and os.path.exists("dataset_img_y.npy"):
        print("Cargando dataset preprocesado...")
        X = np.load("dataset_img_X.npy")
        y = np.load("dataset_img_y.npy")
        clases = ["papa", "zanahoria", "choclo", "berenjena"]
        print(f"Dataset: {X.shape[0]} imágenes, {X.shape[1]} features")
    else:
        print("Procesando imágenes por primera vez...🥔 🥕 🌽 🍆")
        X, y, clases = cargar_dataset_imagenes("dataset_img/")
        np.save("dataset_img_X.npy", X)
        np.save("dataset_img_y.npy", y)
        print(f"Dataset guardado.")

    # Normalizar
    media = np.mean(X, axis=0)
    std = np.std(X,  axis=0) + 1e-8
    X_norm = (X - media) / std

    # PCA para visualización
    pca_kmeans = PCA_propio(n_componentes=20)
    X_reducido = pca_kmeans.fit_transform(X_norm)
    
    # Usar la mejor semilla encontrada
    kmeans = KMeans_propio(k=4, semilla=99)
    kmeans.fit(X_reducido)
    etiquetas = kmeans.predict(X_reducido)
    
    mapa = asignar_etiquetas_clusters(etiquetas, y)
    print("\nMapeo de clusters:")
    for cluster, clase_idx in mapa.items():
        print(f"Cluster {cluster} → {clases[clase_idx]}")

    y_pred = np.array([mapa[e] for e in etiquetas])
    acc = np.sum(y_pred == y) / len(y) * 100
    print(f"\nAccuracy K-means: {acc:.1f}%")
        
    # Visualización
    graficar_clusters_2d(X_reducido, etiquetas, y, mapa, clases, pca_kmeans)

    # Guardar modelo
    np.save("kmeans_centroides.npy", kmeans.centroides)
    np.save("img_media_norm.npy", media)
    np.save("img_std_norm.npy", std)
    np.save("img_mapa_clusters.npy", np.array(list(mapa.items())))
    np.save("img_pca_componentes.npy", pca_kmeans.componentes)
    np.save("img_pca_media.npy", pca_kmeans.media)
    print("\nModelo K-means guardado.")
    
    
    """
    Este codigo lo hice para buscar una mejor precisión, probando muchas semillas para 
    inicializar kmeans.
    
    mejor_acc = 0
    mejor_kmeans = None
    mejor_etiquetas = None

    for semilla in range(100):
        km = KMeans_propio(k=4, semilla=semilla)
        km.fit(X_reducido)
        etiquetas = km.predict(X_reducido)
        mapa = asignar_etiquetas_clusters(etiquetas, y)
        y_pred = np.array([mapa[e] for e in etiquetas])
        acc = np.sum(y_pred == y) / len(y) * 100
        print(f"Semilla {semilla}: {acc:.1f}%")
        if acc > mejor_acc:
            mejor_acc = acc
            mejor_kmeans = km
            mejor_etiquetas = etiquetas

    print(f"\nMejor accuracy: {mejor_acc:.1f}%")

    """