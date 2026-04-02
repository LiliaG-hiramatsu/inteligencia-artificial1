import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from prediccion_voz import KNN, grabar_audio, predecir
from kmeans import extraer_caracteristicas_imagen, KMeans_propio

CLASES = ["papa", "zanahoria", "choclo", "berenjena"]

# ─────────────────────────────────────────────
# 1. CARGAR MODELOS
# ─────────────────────────────────────────────

def cargar_modelo_voz():
    # Modelo general
    X_train     = np.load("knn_X_train.npy")
    y_train     = np.load("knn_y_train.npy")
    componentes = np.load("pca_componentes.npy")
    media_norm  = np.load("norm_media.npy")
    std_norm    = np.load("norm_std.npy")
    pca_media   = np.load("pca_media.npy")
    knn_general = KNN(k=5)
    knn_general.fit(X_train, y_train)

    # Modelo especializado
    X_sub       = np.load("sub_X_train.npy")
    y_sub       = np.load("sub_y_train.npy")
    comp_sub    = np.load("sub_pca_componentes.npy")
    media_sub   = np.load("sub_norm_media.npy")
    std_sub     = np.load("sub_norm_std.npy")
    pca_media_sub = np.load("sub_pca_media.npy")
    knn_sub = KNN(k=5)
    knn_sub.fit(X_sub, y_sub)

    return (knn_general, media_norm, std_norm, pca_media, componentes,
            knn_sub, media_sub, std_sub, pca_media_sub, comp_sub)


def cargar_modelo_imagen():
    centroides  = np.load("kmeans_centroides.npy")
    media_norm  = np.load("img_media_norm.npy")
    std_norm    = np.load("img_std_norm.npy")
    mapa_arr    = np.load("img_mapa_clusters.npy")
    pca_comp    = np.load("img_pca_componentes.npy")
    pca_media   = np.load("img_pca_media.npy")
    mapa        = {int(k): int(v) for k, v in mapa_arr}
    kmeans = KMeans_propio(k=4)
    kmeans.centroides = centroides
    return kmeans, media_norm, std_norm, pca_media, pca_comp, mapa

# ─────────────────────────────────────────────
# 2. CLASIFICAR UNA IMAGEN
# ─────────────────────────────────────────────

def clasificar_imagen(ruta, kmeans, media_norm, std_norm, pca_media, pca_comp, mapa):
    feat      = extraer_caracteristicas_imagen(ruta)
    feat_norm = (feat - media_norm) / (std_norm + 1e-8)
    feat_pca  = (feat_norm - pca_media) @ pca_comp
    cluster   = kmeans.predict(feat_pca.reshape(1, -1))[0]
    clase_idx = mapa[cluster]
    return CLASES[clase_idx]

# ─────────────────────────────────────────────
# 3. CAPTURAR 4 FOTOS CON LA CÁMARA
# ─────────────────────────────────────────────

def capturar_fotos(kmeans, media_norm, std_norm, pca_media, pca_comp, mapa):
    carpeta = "test_img"
    
    # Limpiar carpeta antes de cada ronda
    if os.path.exists(carpeta):
        for f in os.listdir(carpeta):
            os.remove(os.path.join(carpeta, f))
    else:
        os.makedirs(carpeta)

    # Esperar a que el usuario guarde las fotos
    print(f"\n📸 Sacá las fotos con DroidCam y guardalas en '{carpeta}/'")
    print("Cuando termines, presioná ENTER...")
    input()

    # Leer y clasificar las imágenes que haya en la carpeta
    formatos = [".jpg", ".jpeg", ".png"]
    archivos = [f for f in os.listdir(carpeta)
                if any(f.lower().endswith(e) for e in formatos)]

    if not archivos:
        print("❌ No se encontraron imágenes en la carpeta.")
        return []

    fotos = []
    for i, archivo in enumerate(archivos, start=1):
        ruta = os.path.join(carpeta, archivo)
        clase = clasificar_imagen(ruta, kmeans, media_norm, std_norm,
                                pca_media, pca_comp, mapa)
        fotos.append((ruta, clase))
        print(f"  Posición {i}: {archivo} → {clase.upper()}")

    return fotos

# ─────────────────────────────────────────────
# 4. MOSTRAR RESULTADO
# ─────────────────────────────────────────────

def mostrar_resultado(fotos, verdura_solicitada):
    for i, (ruta, clase) in enumerate(fotos):
        if clase == verdura_solicitada:
            img = mpimg.imread(ruta)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{verdura_solicitada.upper()}  —  Posición: {i + 1}",
                         fontsize=16, fontweight="bold", color="green")
            plt.tight_layout()
            plt.show(block=False)
            
            print(f"\n{'='*40}")
            print(f"  Verdura: {verdura_solicitada.upper()}")
            print(f"  Posición: {i + 1}")
            print(f"{'='*40}")
            
            opcion = input("ENTER = otra verdura  |  Q + ENTER = salir: ").strip().lower()
            plt.close()
            return "salir" if opcion == "q" else "continuar"
    
    print(f"\n⚠️  '{verdura_solicitada.upper()}' no fue identificada en ninguna foto.")
    opcion = input("ENTER = continuar  |  Q + ENTER = salir: ").strip().lower()
    return "salir" if opcion == "q" else "continuar"

# ─────────────────────────────────────────────
# 5. LOOP PRINCIPAL DEL AGENTE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Cargando modelos...")
    (knn_general, media_voz, std_voz, pca_media_voz, comp_voz,
    knn_sub, media_sub, std_sub, pca_media_sub, comp_sub) = cargar_modelo_voz()
    kmeans, media_img, std_img, pca_media_img, pca_comp_img, mapa = cargar_modelo_imagen()
    print("✅ Modelos cargados.")

    print("\n" + "="*40)
    print("   AGENTE - CLASIFICADOR DE VERDURAS")
    print("   Palabras: papa | zanahoria | choclo | berenjena")
    print("="*40)

    while True:
        # ── Paso 1: reconocimiento de voz ──
        print("\nPresioná ENTER para nombrar una verdura (Q + ENTER para salir):")
        opcion = input().strip().lower()

        if opcion == 'q':
            print("¡Hasta luego!")
            break

        try:
            archivo = grabar_audio(duracion=2)
            verdura_solicitada, confianza = predecir(
                archivo, knn_general, knn_sub,
                media_voz, std_voz, pca_media_voz, comp_voz,
                media_sub, std_sub, pca_media_sub, comp_sub
            )
            os.remove(archivo)
        except Exception as e:
            print(f"❌ Error en reconocimiento de voz: {e}")
            continue

        # ── Paso 2: capturar 4 fotos y clasificarlas ──
        print(f"\nVerdura solicitada: {verdura_solicitada.upper()}")
        print("Abriendo cámara... Mostrá cada verdura y presioná ESPACIO.")
        fotos = capturar_fotos(kmeans, media_img, std_img,
                               pca_media_img, pca_comp_img, mapa)

        if not fotos:
            continue  # captura cancelada

        # ── Paso 3: mostrar resultado ──
        accion = mostrar_resultado(fotos, verdura_solicitada)

        if accion == "salir":
            print("¡Hasta luego!")
            break
        # si accion == "continuar" → vuelve al inicio del while