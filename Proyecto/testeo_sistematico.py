import os
import librosa
import numpy as np
import csv
from collections import Counter

# CONFIGURACIÓN
ruta_dataset = 'mfcc_audio_dataset.csv'
ruta_test = 'test_audio'
n_mfcc = 13
k = 5

# FUNCIONES
def cargar_dataset(ruta_csv):
    X = []
    y = []
    with open(ruta_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Saltar encabezado
        for fila in reader:
            caracteristicas = list(map(float, fila[:-1]))
            etiqueta = fila[-1]
            X.append(caracteristicas)
            y.append(etiqueta)
    return np.array(X), np.array(y)

def extraer_vector(archivo_audio):
    y, sr = librosa.load(archivo_audio, sr=None)
    y, _ = librosa.effects.trim(y)  # Recortar silencios
    y = librosa.util.normalize(y)   # Normalizar volumen
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate((mfcc_mean, mfcc_std))  # 26 características

def distancia_euclidea(a, b):
    return np.linalg.norm(a - b)

def knn(X_train, y_train, x_test, k):
    distancias = [distancia_euclidea(x_test, x) for x in X_train]
    indices_ordenados = np.argsort(distancias)
    vecinos = y_train[indices_ordenados[:k]]
    voto = Counter(vecinos).most_common(1)[0][0]
    return voto

# Cargar datos de entrenamiento
X_train, y_train = cargar_dataset(ruta_dataset)

# Evaluar cada audio de test
print("📋 Evaluación del reconocimiento de voz:\n")
correctas = 0
total = 0

for archivo in os.listdir(ruta_test):
    if archivo.endswith(".wav"):
        ruta = os.path.join(ruta_test, archivo)
        esperado = archivo.split('_')[0].lower()  # extrae 'papa', 'choclo', etc.

        try:
            vector = extraer_vector(ruta)
            predicho = knn(X_train, y_train, vector, k)
            resultado = "✅" if predicho == esperado else "❌"
            print(f"{archivo}: {resultado} esperado={esperado} | predicho={predicho}")
            if predicho == esperado:
                correctas += 1
            total += 1
        except Exception as e:
            print(f"Error con {archivo}: {e}")

# Mostrar precisión
if total > 0:
    precision = correctas / total * 100
    print(f"\n🎯 Precisión total: {precision:.2f}% ({correctas}/{total})")
else:
    print("⚠️ No se encontraron audios de prueba.")
