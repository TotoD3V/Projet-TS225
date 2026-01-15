import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def homography_estimate(x1, y1, x2, y2):
    A = np.array([[x1[0],y1[0],1,0,0,0,-x2[0]*x1[0],-x2[0]*y1[0]],
                  [0,0,0,x1[0],y1[0],1,-x1[0]*y2[0],-y1[0]*y2[0]],
                  [x1[1],y1[1],1,0,0,0,-x2[1]*x1[1],-x2[1]*y1[1]],
                  [0,0,0,x1[1],y1[1],1,-x1[1]*y2[1],-y1[1]*y2[1]],
                  [x1[2],y1[2],1,0,0,0,-x2[2]*x1[2],-x2[2]*y1[2]],
                  [0,0,0,x1[2],y1[2],1,-x1[2]*y2[2],-y1[2]*y2[2]],
                  [x1[3],y1[3],1,0,0,0,-x2[3]*x1[3],-x2[3]*y1[3]],
                  [0,0,0,x1[3],y1[3],1,-x1[3]*y2[3],-y1[3]*y2[3]]])
    b = np.array([x2[0],y2[0],x2[1],y2[1],x2[2],y2[2],x2[3],y2[3]])
    solution = np.linalg.solve(A,b)
    print(solution)
    solution_f = np.array([[solution[0], solution[1], solution[2]],
                           [solution[3], solution[4], solution[5]],
                           [solution[6], solution[7],     1      ]])
    print(solution_f)
    return solution_f

def homography_apply(H, x1, y1):
    denom = H[2,0] * x1 + H[2,1] * y1 + H[2,2]
    x2 = (H[0,0] * x1 + H[0,1] * y1 + H[0,2]) / denom
    y2 = (H[1,0] * x1 + H[1,1] * y1 + H[1,2]) / denom
    return (x2, y2)

def homography_extraction(I1, x, y, w, h):
    if (len(I1.shape) == 2):
        I2 = np.zeros((w,h))
        xr = np.array([0,w,w,0])
        yr = np.array([0,0,h,h])
        H = homography_estimate(xr, x, yr, y)
        for i in range(w):
            for j in range(h):
                xf, yf = homography_apply(H, i, j)
                xf, yf = int(xf), int(yf)
                # Vérification des limites
                if 0 <= xf < I1.shape[1] and 0 <= yf < I1.shape[0]:
                    I2[i, j] = I1[yf, xf]
        return I2
    elif (len(I1.shape) == 3):
        I2 = np.zeros((h, w, 3)).astype("uint8")
        xr = np.array([0, w-1, w-1, 0])
        yr = np.array([0, 0, h-1, h-1])  
        H = homography_estimate(xr, yr, x, y)
        for i in range(w):
            for j in range(h):
                xf, yf = map(int, homography_apply(H, i, j))
                # Vérification des limites
                if 0 <= xf < I1.shape[1] and 0 <= yf < I1.shape[0]:
                    I2[j, i, :] = I1[yf, xf, :]     
        return I2
import numpy as np

def homography_projection(I1, I2, x, y):
    """
    Projette l'image source I1 (rectangulaire) dans le quadrilatère (x, y) de I2.
    """
    # 1. Dimensions de l'image source
    h1, w1 = I1.shape[:2]
    
    # 2. Coordonnées des 4 coins de l'image source (rectangle)
    # Ordre : Haut-Gauche, Haut-Droite, Bas-Droite, Bas-Gauche
    xsrc = np.array([0, w1 - 1, w1 - 1, 0])
    ysrc = np.array([h1 - 1, h1 - 1, 0, 0])
    
    # 3. Identification de l'homographie : Source -> Destination 
    H_src_to_dst = homography_estimate(xsrc, ysrc, x, y)
    
    # 4. Pour le remplissage (balayage), on a besoin de l'inverse : Destination -> Source
    # Cela permet d'éviter les trous dans l'image finale.
    det_H = np.linalg.det(H_src_to_dst)
    if det_H != 0:
        H_dst_to_src = np.linalg.inv(H_src_to_dst)
    else:
        print("Erreur : Matrice d'homographie non inversible.")
        return I2

    # 5. Déterminer la boîte englobante du quadrilatère dans I2 pour limiter les calculs
    min_x, max_x = int(np.floor(min(x))), int(np.ceil(max(x)))
    min_y, max_y = int(np.floor(min(y))), int(np.ceil(max(y)))
    
    # Sécurité pour ne pas sortir de I2
    min_x, max_x = max(0, min_x), min(I2.shape[1] - 1, max_x)
    min_y, max_y = max(0, min_y), min(I2.shape[0] - 1, max_y)

    # 6. Balayage de la zone de destination
    for px in range(min_x, max_x + 1):
        for py in range(min_y, max_y + 1):
            # Appliquer l'homographie inverse pour trouver le point source (sx, sy)
            sx, sy = homography_apply(H_dst_to_src, px, py)
            
            # Vérifier si le point calculé tombe dans le domaine de validité de I1
            if 0 <= sx < w1 and 0 <= sy < h1:
                # Remplacement du contenu de I2 par celui de I1 (échantillonnage plus proche voisin)
                I2[py, px, :] = I1[int(sy), int(sx), :]
                
    return I2
def get_bbox(x, y):
        return int(np.floor(min(x))), int(np.ceil(max(x))), int(np.floor(min(y))), int(np.ceil(max(y)))

def homography_cross_projection(I, x1, y1, x2, y2):
    """
    Échange les contenus des quadrilatères (x1, y1) et (x2, y2) dans l'image I.
    Utilise une approche directe (une seule passe) pour une qualité optimale.
    """
    # 1. Créer une copie pour ne pas perdre les données source lors de l'écriture
    I_res = I.copy()
    
    # 2. Identifier les homographies directes entre les deux quadrilatères
    # H12 : permet de trouver le point dans le quad 2 correspondant à un point du quad 1
    H12 = homography_estimate(x1, y1, x2, y2)
    # H21 : permet de trouver le point dans le quad 1 correspondant à un point du quad 2
    H21 = homography_estimate(x2, y2, x1, y1)
    

    min_x1, max_x1, min_y1, max_y1 = get_bbox(x1, y1)
    min_x2, max_x2, min_y2, max_y2 = get_bbox(x2, y2)

    # 4. PASSE UNIQUE : Remplissage de la zone 1 avec les pixels de la zone 2
    for px in range(min_x1, max_x1 + 1):
        for py in range(min_y1, max_y1 + 1):
            # On cherche la source dans le quad 2
            sx, sy = homography_apply(H12, px, py)
            # Vérification si le point est bien dans le quadrilatère source (zone 2)
            if 0 <= sx < I.shape[1] and 0 <= sy < I.shape[0]:
                I_res[py, px, :] = I[int(sy), int(sx), :]

    # 5. PASSE UNIQUE : Remplissage de la zone 2 avec les pixels de la zone 1
    for px in range(min_x2, max_x2 + 1):
        for py in range(min_y2, max_y2 + 1):
            # On cherche la source dans le quad 1
            sx, sy = homography_apply(H21, px, py)
            if 0 <= sx < I.shape[1] and 0 <= sy < I.shape[0]:
                I_res[py, px, :] = I[int(sy), int(sx), :]

    return I_res

if __name__ == "__main__":
    # 1. Charger l'image
    img_color = Image.open("../img/challenge1.png").convert("RGB")
    I1 = np.asarray(img_color).astype("uint8")

    # 2. Coordonnées approximatives du carré ROUGE (x=horizontal, y=vertical)
    # Ordre : Haut-Gauche, Haut-Droite, Bas-Droite, Bas-Gauche
    x_rouge = np.array([375,520,507,360])
    y_rouge = np.array([25, 105, 260, 200])

    # 3. Paramètres de sortie (largeur w, hauteur h désirées)
    largeur_dest = 500
    hauteur_dest = 500

    I2 = homography_extraction(I1, x_rouge, y_rouge, largeur_dest, hauteur_dest)
    
    # 4. Sauvegarder l'image I2 dans le dossier img
    plt.imsave("../img/carre_rouge_extrait.jpg", I2)
    # 5. Affichage du résultat
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image Originale")
    plt.imshow(I1)
    plt.subplot(1, 2, 2)
    plt.title("Carré Rouge")
    plt.imshow(I2)
    plt.show()
    # 1. Charger l'image du mur (Destination)
    I_dest = np.asarray(img_color).astype("uint8")
    I_rouge = plt.imread("../img/carre_rouge_extrait.jpg")

    # 3. Coordonnées du carré VERT dans l'image destination (x=horizontal, y=vertical)
    x_vert = [794, 876, 854, 770] 
    y_vert = [175, 250, 380, 320]

    # 4. Appliquer la projection
    # On place le contenu de I_rouge dans la zone (x_vert, y_vert) de I_dest
    I_finale = homography_projection(I_rouge, I_dest, x_vert, y_vert)

    # 5. Affichage
    plt.imshow(I_finale)
    plt.title("Projection du carré rouge sur le carré vert")
    plt.show()

    # Exécuter l'échange
    I_echange = homography_cross_projection(I1, x_rouge, y_rouge, x_vert, y_vert)

    # Affichage
    plt.imshow(I_echange)
    plt.title("Échange des contenus Rouge et Vert")
    plt.show()