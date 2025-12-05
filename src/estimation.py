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
    if (len(I1.shape) == 2):
        I2 = np.zeros((w,h))
        xr = np.array([0,w,w,0])
        yr = np.array([0,0,h,h])
        H = homography_estimate(xr, x, yr, y)
        for i in range(w):
            for j in range(h):
                xf, yf = homography_apply(H, i, j)
                I2[i, j] = I1[int(xf), int(yf)]
        return I2
    elif (len(I1.shape) == 3):
        I2 = np.zeros((w,h,3)).astype("uint8")
        xr = np.array([0,w,w,0])
        yr = np.array([0,0,h,h])
        H = homography_estimate(xr, yr, x, y)
        for i in range(w):
            for j in range(h):
                xf, yf = map(int, homography_apply(H, i, j))
                I2[j, i, :] = I1[yf, xf, :]     
        return I2

if __name__ == "__main__":
    Immeuble = Image.open("../img/mbappe.jpg")
    I1 = np.asarray(Immeuble).astype("uint8")
    x = [634, 909, 962, 650]
    y = [92, 117, 350, 338]
    I2 = homography_extraction(I1, x, y, 500, 500)
    plt.imsave("../img/MBappe.jpg", I2)