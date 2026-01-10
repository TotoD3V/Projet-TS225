#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 09:50:10 2025

@author: thomas
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

def homography_extraction(I1, x, y, w, h):
    I2 = np.zeros((h, w, I1.shape[2]), dtype=I1.dtype)
    xr = np.array([0,w-1,w-1,0])
    yr = np.array([0,0,h-1,h-1])
    H = homography_estimate(xr, yr, x, y)
    for i in range(h):
        for j in range(w):
            xs, ys = homography_apply(H, np.array([j]), np.array([i]))
            
            xs = int(round(xs[0]))
            ys = int(round(ys[0]))
            
            if 0 <= xs < I1.shape[1] and 0 <= ys < I1.shape[0]:
                I2[i, j] = I1[ys, xs]
    return I2, H

quadrangle = []
quadrangle.append({"coords" : [[375, 518, 507, 360], [28, 104, 258, 202]],
         "aire" : 0, "couleur" : "rouge"})
quadrangle.append({"coords" : [[310, 498, 553, 385], [341, 300, 498, 569]],
         "aire" : 0, "couleur" : "violet"})
quadrangle.append({"coords" : [[423, 551, 465, 313], [570, 669, 837, 752]],
         "aire" : 0, "couleur" : "bleu"})
quadrangle.append({"coords" : [[591, 723, 742, 617], [117, 144, 325, 327]],
         "aire" : 0, "couleur" : "jaune"})
quadrangle.append({"coords" : [[648, 736, 697, 600], [347, 424, 551, 484]],
        "aire" : 0, "couleur" : "rose"})
quadrangle.append({"coords" : [[596, 705, 739, 638], [631, 568, 709, 786]],
        "aire" : 0, "couleur": "gris"})
quadrangle.append({"coords" : [[793, 874, 854, 768], [175, 251, 383, 321]],
        "aire" : 0, "couleur" : "vert"})
quadrangle.append({"coords" : [[835, 863, 813, 781], [397, 490, 539, 442]],
        "aire" : 0, "couleur" : "pale"})
quadrangle.append({"coords" : [[812, 868, 837, 775], [571, 625, 729, 682]],
        "aire" : 0, "couleur" : "noir"})
img = {"coords" : [[364, 894, 873, 286], [5, 235, 749, 904]],
        "aire" : 0}

I1 = plt.imread("./img/challenge1.png")
I2, H = homography_extraction(I1, img["coords"][0], img["coords"][1], 1500, 1000)

for i in range(9):
    x2, y2 = homography_apply(H, np.array(quadrangle[i]["coords"][0]), np.array(quadrangle[i]["coords"][1]))
    quadrangle[i]["aire"] = (1/2) * abs(x2[0] * y2[1] + x2[1] * y2[2] + x2[2] * y2[3] + x2[3] * y2[0] - (y2[0] * x2[1] + y2[1] * x2[2] + y2[2] * x2[3] + y2[3] * x2[0]))

aire_max = max([quadrangle[i]["aire"] for i in range(9)])


for i in range(9):
    quadrangle[i]["aire"] /= aire_max

aires = [(quadrangle[i]["couleur"], quadrangle[i]["aire"]) for i in range(9)]
aires.sort(key=lambda x: x[1], reverse=True)

for i in range(9):  
    print("Aire {} : {}".format(aires[i][0], aires[i][1]))