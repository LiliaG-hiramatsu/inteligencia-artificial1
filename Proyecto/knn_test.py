import numpy as np
import csv
import librosa
from collections import Counter

# -------- CONFIGURACIÓN --------
archivo_dataset = 'mfcc_audio_dataset.csv'
archivo_audio = 'audio_test.wav'  # el nuevo archivo a clasificar (zanahoria)
n_mfcc = 13
k = 5  # número de vecinos

# -------- FUNCIONES --------

def cargar_dataset(ruta_csv):
    X = []
    y = []
    with open(ruta_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # saltar encabezado
        for fila in reader:
            caracteristicas = list(map(float, fila[:-1]))  # las 26 columnas
            etiqueta = fila[-1]
            X.append(caracteristicas)
            y.append(etiqueta)
    return np.array(X), np.array(y)

def extraer_mfcc(archivo):
    y, sr = librosa.load(archivo, sr=None)
    y, _ = librosa.effects.trim(y)  # elimina silencios al inicio/final
    y = librosa.util.normalize(y) # normaliza el volumen
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate((mfcc_mean, mfcc_std))  # vector de 26 elementos

def distancia_euclidea(a, b):
    return np.linalg.norm(a - b)

def knn(X_train, y_train, x_test, k):
    distancias = [distancia_euclidea(x_test, x) for x in X_train]
    indices_ordenados = np.argsort(distancias)
    vecinos = y_train[indices_ordenados[:k]]
    voto = Counter(vecinos).most_common(1)[0][0]
    return voto

# -------- FLUJO PRINCIPAL --------

# Paso 1: cargar la base de datos
X_train, y_train = cargar_dataset(archivo_dataset)

# Paso 2: procesar nuevo audio
x_nuevo = extraer_mfcc(archivo_audio)

# Paso 3: aplicar KNN
prediccion = knn(X_train, y_train, x_nuevo, k)

print(f"🔊 Palabra reconocida: {prediccion}")
