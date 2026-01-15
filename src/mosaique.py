#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 14:33:02 2026

@author: thomas
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class MIB:
    masque: np.ndarray   # masque binaire (h, w)
    image: np.ndarray    # image extraite (h, w, c)
    boite: np.ndarray    # quadrilatère 4x2
    
def homography_estimate(x1, y1, x2, y2):
    H = np.array([[x1[0],y1[0],1,0,0,0,-x2[0]*x1[0],-x2[0]*y1[0]],
                  [0,0,0,x1[0],y1[0],1,-x1[0]*y2[0],-y1[0]*y2[0]],
                  [x1[1],y1[1],1,0,0,0,-x2[1]*x1[1],-x2[1]*y1[1]],
                  [0,0,0,x1[1],y1[1],1,-x1[1]*y2[1],-y1[1]*y2[1]],
                  [x1[2],y1[2],1,0,0,0,-x2[2]*x1[2],-x2[2]*y1[2]],
                  [0,0,0,x1[2],y1[2],1,-x1[2]*y2[2],-y1[2]*y2[2]],
                  [x1[3],y1[3],1,0,0,0,-x2[3]*x1[3],-x2[3]*y1[3]],
                  [0,0,0,x1[3],y1[3],1,-x1[3]*y2[3],-y1[3]*y2[3]]])
    b = np.array([x2[0],y2[0],x2[1],y2[1],x2[2],y2[2],x2[3],y2[3]])
    solution = np.linalg.solve(H,b)
    solution_f = np.array([[solution[0], solution[1], solution[2]],
                           [solution[3], solution[4], solution[5]],
                           [solution[6], solution[7],     1      ]])
    return solution_f

def homography_apply(H, x1, y1):
    denom = H[2,0] * x1 + H[2,1] * y1 + H[2,2]
    x2 = (H[0,0] * x1 + H[0,1] * y1 + H[0,2]) / denom
    y2 = (H[1,0] * x1 + H[1,1] * y1 + H[1,2]) / denom
    return (x2, y2)

def I_to_mib(I, x, y, w, h):
    """
    Extrait un bloc rectangulaire axis-aligned de l'image I,
    défini par son coin haut-gauche (x, y) et sa taille (w, h).
    Renvoie un MIB contenant :
        - l'image extraite
        - un masque plein (tout à 1)
        - la boîte englobante (x, y, w, h)
    """

    # Sécurisation des bornes
    H, W = I.shape[:2]
    x2 = min(x + w, W)
    y2 = min(y + h, H)

    # Extraction de la sous-image
    I2 = I[y:y2, x:x2].copy()

    # Dimensions réelles extraites (si la boîte dépasse l'image)
    h2 = I2.shape[0]
    w2 = I2.shape[1]

    # Masque plein
    M = np.ones((h2, w2), dtype=np.uint8)

    # Boîte englobante dans le repère global
    boite = (x, y, w2, h2)

    return MIB(masque=M, image=I2, boite=boite)

def mib_transform(mib, H):
    """
    Applique une homographie H à un MIB.
    Transforme l'image, le masque et met à jour la boîte.
    """

    I = mib.image
    M = mib.masque
    x0, y0, w, h = mib.boite

    # 1. Coordonnées des 4 coins du rectangle local
    pts = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    # 2. Transformation des coins pour obtenir la nouvelle boîte
    pts_t = np.array([homography_apply(H, px + x0, py + y0) for px, py in pts])
    xs = pts_t[:, 0]
    ys = pts_t[:, 1]

    # Nouvelle boîte englobante
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    new_w = xmax - xmin
    new_h = ymax - ymin

    # 3. Création des nouvelles images
    I2 = np.zeros((new_h, new_w, I.shape[2]), dtype=np.uint8)
    M2 = np.zeros((new_h, new_w), dtype=np.uint8)

    # 4. Remplissage par homographie inverse
    for i in range(new_w):
        for j in range(new_h):
            # Coordonnées globales
            Xg = xmin + i
            Yg = ymin + j

            # Coordonnées dans l'ancien MIB
            x_old, y_old = homography_apply(np.linalg.inv(H), Xg, Yg)

            # Coordonnées locales dans l'image du MIB
            x_local = x_old - x0
            y_local = y_old - y0

            if 0 <= x_local < w and 0 <= y_local < h:
                xi, yi = int(x_local), int(y_local)
                if M[yi, xi] == 1:
                    I2[j, i, :] = I[yi, xi, :]
                    M2[j, i] = 1

    # 5. Retour du nouveau MIB
    return MIB(
        masque=M2,
        image=I2,
        boite=(xmin, ymin, new_w, new_h)
    )

def mib_fusion(canvas, mib):
    """
    Fusionne un MIB dans un canevas global.
    Le masque du MIB détermine quels pixels sont copiés.
    """

    I2 = mib.image
    M  = mib.masque
    x, y, w, h = mib.boite

    Hc, Wc = canvas.shape[:2]

    # Limites réelles dans le canevas (sécurisation)
    x_end = min(x + w, Wc)
    y_end = min(y + h, Hc)

    # Limites réelles dans le MIB (si coupé par les bords)
    w2 = x_end - x
    h2 = y_end - y

    # Fusion pixel par pixel
    for j in range(h2):
        for i in range(w2):
            if M[j, i] == 1:
                canvas[y + j, x + i, :] = I2[j, i, :]

    return canvas
