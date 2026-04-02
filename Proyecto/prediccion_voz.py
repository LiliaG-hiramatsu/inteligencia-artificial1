import numpy as np
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os
from collections import Counter

clases = ["papa", "zanahoria", "choclo", "berenjena"]

# ─────────────────────────────────────────────
# 1. CARGAR MODELO ENTRENADO
# ─────────────────────────────────────────────

def cargar_modelo():
    X_train = np.load("knn_X_train.npy")
    y_train = np.load("knn_y_train.npy")
    componentes = np.load("pca_componentes.npy")
    media_norm = np.load("norm_media.npy")
    std_norm = np.load("norm_std.npy")
    pca_media = np.load("pca_media.npy")
    return X_train, y_train, componentes, media_norm, std_norm, pca_media

# modelo sin tener en cuenta papa
def cargar_modelo_especializado():
    X_train      = np.load("sub_X_train.npy")
    y_train      = np.load("sub_y_train.npy")
    componentes  = np.load("sub_pca_componentes.npy")
    media_norm   = np.load("sub_norm_media.npy")
    std_norm     = np.load("sub_norm_std.npy")
    pca_media    = np.load("sub_pca_media.npy")
    return X_train, y_train, componentes, media_norm, std_norm, pca_media

# ─────────────────────
# 2. KNN
# ─────────────────────

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _distancia_euclidiana(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predecir_uno(self, x):
        distancias = [self._distancia_euclidiana(x, xt)
                      for xt in self.X_train]
        indices_k  = np.argsort(distancias)[:self.k]
        etiquetas_k = [self.y_train[i] for i in indices_k]
        votos = Counter(etiquetas_k)
        return votos.most_common(1)[0][0], votos


# ─────────────────────────────────────────────
# 3. PIPELINE: AUDIO → FEATURES → PCA → KNN
# ─────────────────────────────────────────────

def extraer_caracteristicas(ruta_audio, n_mfcc=13):
    y, sr = librosa.load(ruta_audio, sr=None)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs,  axis=1)

    delta = librosa.feature.delta(mfccs)
    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta,  axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    return np.concatenate([
        mfccs_mean, mfccs_std,
        delta_mean, delta_std,
        [zcr, rms, centroid, rolloff]
    ])


def preprocesar_audio(features_raw, media_norm, std_norm, pca_media, componentes):
    # Normalizar (igual que en entrenamiento)
    features_norm = (features_raw - media_norm) / (std_norm + 1e-8)

    # 3. PCA con la media del entrenamiento (no del audio individual)
    features_centrado = features_norm - pca_media
    return features_centrado @ componentes


# ─────────────────────────────────────────────
# 4. GRABACIÓN DESDE MICRÓFONO
# ─────────────────────────────────────────────

def grabar_audio(duracion=2, sample_rate=16000):
    """
    Graba audio del micrófono por 'duracion' segundos
    """
    print(f"\n🎙️ Grabando {duracion} segundos... ¡Hablá ahora!")
    audio = sd.rec(
        int(duracion * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Esperar a que termine la grabación
    print("✅Grabación finalizada")

    # Guardar temporalmente como .wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        archivo_temp = f.name
        
    wav.write(archivo_temp, sample_rate, audio)
    return archivo_temp


# ─────────────────────────────────────────────
# 5. PREDICCIÓN JERÁRQUICA
# ─────────────────────────────────────────────

def predecir(archivo_audio, knn_general, knn_sub,
             media_norm, std_norm, pca_media, componentes,
             media_sub, std_sub, pca_media_sub, componentes_sub):

    # Extraer features raw una sola vez
    features_raw = extraer_caracteristicas(archivo_audio)

    # ── Clasificador general ──
    feat_general = preprocesar_audio(features_raw, media_norm, std_norm, pca_media, componentes)
    clase_general, votos_general = knn_general.predecir_uno(feat_general)

    # ── Si predice papa, resultado final ──
    if clase_general == 0:
        confianza = votos_general[0] / sum(votos_general.values()) * 100
        print("\n" + "="*40)
        print(f"Predicción: PAPA")
        print(f"Confianza:  {confianza:.0f}%  ({votos_general[0]}/{sum(votos_general.values())} votos)")
        print(f"[Clasificador: general]")
        print("="*40)
        return "papa", confianza

    # ── Si no, usar sub-clasificador ──
    feat_sub = preprocesar_audio(features_raw, media_sub, std_sub, pca_media_sub, componentes_sub)
    clase_sub, votos_sub = knn_sub.predecir_uno(feat_sub)

    clase_predicha = clases[clase_sub]
    confianza      = votos_sub[clase_sub] / sum(votos_sub.values()) * 100

    print("\n" + "="*40)
    print(f"Predicción: {clase_predicha.upper()}")
    print(f"Confianza:  {confianza:.0f}%  ({votos_sub[clase_sub]}/{sum(votos_sub.values())} votos)")
    print(f"[Clasificador: general → especializado]")
    print("="*40)
    
    return clase_predicha, confianza


# ─────────────────────────────────────────────
# 6. LOOP PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Cargando modelo...")
    X_train, y_train, componentes, media_norm, std_norm, pca_media = cargar_modelo()
    X_sub, y_sub, comp_sub, media_sub, std_sub, pca_media_sub = cargar_modelo_especializado()

    knn_general = KNN(k=5)
    knn_general.fit(X_train, y_train)

    knn_sub = KNN(k=5)
    knn_sub.fit(X_sub, y_sub)
    
    print("Modelos listos")

    print("\n" + "="*40)
    print("RECONOCEDOR DE VERDURAS POR VOZ")
    print("Palabras: papa | zanahoria | choclo | berenjena")
    print("="*40)

    while True:
        print("\nOpciones:")
        print("[Enter] → Grabar desde micrófono")
        print("[a] → Usar archivo .wav existente")
        print("[q] → Salir")

        opcion = input("\nElegí una opción: ").strip().lower()

        if opcion == "q":
            print("¡Hasta luego!")
            break

        elif opcion == "a":
            ruta = input("Ruta del archivo .wav: ").strip()
            if os.path.exists(ruta):
                predecir(ruta, knn_general, knn_sub,
                         media_norm, std_norm, pca_media, componentes,
                         media_sub, std_sub, pca_media_sub, comp_sub)
            else:
                print("❌Archivo no encontrado")

        else:
            # Grabar desde micrófono
            try:
                archivo = grabar_audio(duracion=2)
                predecir(archivo, knn_general, knn_sub,
                         media_norm, std_norm, pca_media, componentes,
                         media_sub, std_sub, pca_media_sub, comp_sub)
                os.remove(archivo)  # limpiar archivo temporal
            except Exception as e:
                print(f"❌Error: {e}")
