import os
import librosa
import numpy as np
import csv

# Ruta a la carpeta principal del dataset
ruta_dataset = 'dataset_audio'

# Configuración de MFCC
n_mfcc = 13  # Número de coeficientes a extraer

# Lista para guardar todos los vectores y etiquetas
data = []

# Recorrer cada subcarpeta (una por clase)
for etiqueta in os.listdir(ruta_dataset):
    carpeta_clase = os.path.join(ruta_dataset, etiqueta)
    
    if not os.path.isdir(carpeta_clase):
        continue

    for archivo in os.listdir(carpeta_clase):
        if archivo.endswith(".wav"):
            ruta_audio = os.path.join(carpeta_clase, archivo)
            try:
                y, sr = librosa.load(ruta_audio, sr=None)
                y, _ = librosa.effects.trim(y)  # elimina silencios al inicio/final
                y = librosa.util.normalize(y) # normaliza el volumen
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                vector_caracteristicas = np.concatenate((mfcc_mean, mfcc_std))  # 26 características

                # Agregar etiqueta al final
                fila = vector_caracteristicas.tolist() + [etiqueta]
                data.append(fila)

            except Exception as e:
                print(f"Error al procesar {ruta_audio}: {e}")

# Guardar los datos en un CSV
with open('mfcc_audio_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    encabezado = [f'c{i}' for i in range(26)] + ['etiqueta']
    writer.writerow(encabezado)
    writer.writerows(data)

print("Extracción completa. Archivo guardado como mfcc_audio_dataset.csv")
