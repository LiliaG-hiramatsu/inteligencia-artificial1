import os
from pydub import AudioSegment

def convertir_a_wav(carpeta_entrada, carpeta_salida):
    """
    Convierte todos los audios (.ogg, .mp3, .m4a, .opus)
    de una carpeta a carpeta_entrada y los guarda en carpeta carpeta_salida
    """
    formatos = [".ogg", ".mp3", ".m4a", ".opus", ".mp4"]
    os.makedirs(carpeta_salida, exist_ok=True)

    archivos = [f for f in os.listdir(carpeta_entrada)
                if any(f.endswith(ext) for ext in formatos)]

    if not archivos:
        print("No se encontraron archivos para convertir.")
        return

    for archivo in archivos:
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        nombre_sin_ext = os.path.splitext(archivo)[0]
        ruta_salida = os.path.join(carpeta_salida, nombre_sin_ext + ".wav")

        try:
            ext = os.path.splitext(archivo)[1].replace(".", "")
            audio = AudioSegment.from_file(ruta_entrada, format=ext)

            # Estandarizar: mono, 16kHz (ideal para voz)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            audio.export(ruta_salida, format="wav")
            print(f"✓ {archivo} → {nombre_sin_ext}.wav")

        except Exception as e:
            print(f"✗ Error en {archivo}: {e}")

    print(f"\n✓ Conversión completa. Archivos en: {carpeta_salida}")

verduras = ["papa", "zanahoria", "choclo", "berenjena"]

for verdura in verduras:
    entrada = f"audios/{verdura}"
    salida  = f"dataset_audio/{verdura}"

    if os.path.exists(entrada):
        print(f"\n── Convirtiendo: {verdura} ──")
        convertir_a_wav(entrada, salida)
    else:
        print(f"⚠ Carpeta no encontrada: {entrada}")
