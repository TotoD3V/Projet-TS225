import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def homography_estimate(x1,x2,y1,y2):
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
    denom = H[3,1] * x1 + H[3,2] * y1 + H[3,3]
    x2 = (H[1,1] * x1 + H[1,2] * y1 + H[1,3]) / denom
    y2 = (H[2,1] * x1 + H[2,2] * y2 + H[2,3]) / denom
    return (x2, y2)

def homography_extraction(I1, x, y, w, h):
    if (len(I1.shape) == 2):
        I2 = np.zeros((w,h))
        xr = np.array([0,w,w,0])
        yr = np.array([0,0,h,h])
        H = homography_estimate(xr, yr, x, y)
        for i in range(w):
            for j in range(h):
                xf, yf = homography_apply(H, i, j)
                I2[i, j] = I1[int(xf), int(yf)]
    elif (len(I1.shape) == 3):
        I2 = np.zeros((w,h,3))
        xr = np.array([0,w,w,0])
        yr = np.array([0,0,h,h])
        H = homography_estimate(xr, yr, x, y)
        for i in range(w):
            for j in range(h):
                xf, yf = homography_apply(H, i, j)
                I2[i, j, :] = I1[int(xf), int(yf), :]

if __name__ == "main":
    Immeuble = Image.open("./img/mbappe.jpg")
    plt.imshow(Immeuble)
    # I2 = homography_extraction(Immeuble, 500, 500)
    # MBappe = Image.fromarray()