import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

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

# Prueba visual de las 4 clases
verduras = {
    "papa":      "dataset_img/papa/20241022_110339.jpg",
    "zanahoria": "dataset_img/zanahoria/IMG_20260322_130115686.jpg",
    "choclo":    "dataset_img/choclo/IMG_20260322_125725258.jpg",
    "berenjena": "dataset_img/berenjena/IMG_20260322_125936309.jpg",
}

fig, axes = plt.subplots(4, 2, figsize=(8, 16))

for i, (nombre, ruta) in enumerate(verduras.items()):
    img = Image.open(ruta).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    
    h, w  = arr.shape[:2]
    mh, mw = int(h * 0.2), int(w * 0.2)

    resultado, mask2 = aplicar_grabcut(arr)
    
    # Recorte central
    centro = resultado[mh:h-mh, mw:w-mw, :]
    mask_centro = mask2[mh:h-mh, mw:w-mw]

    feat_completa = histogramas_hsv(resultado, mask=mask2)
    feat_centro   = histogramas_hsv(centro, mask=mask_centro)

    axes[i, 0].imshow(arr.astype(np.uint8))
    axes[i, 0].set_title(f"{nombre} - Original")
    axes[i, 1].imshow(resultado.astype(np.uint8))
    axes[i, 1].set_title(f"{nombre} - GrabCut")

plt.tight_layout()
plt.show()




