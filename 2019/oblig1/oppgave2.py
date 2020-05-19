import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.signal import convolve2d


def main():
    img = imread('bilder/cellekjerner.png', flatten=True)
    kernel = gaussian(3)#(1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])
    blurred_img = convolve(img, kernel)
    result = canny(blurred_img)
    plt.imsave('bilder/result.png', result, cmap='gray')


def canny(img):
    Gx = convolve(img, np.array([[1,0,-1],[2,0,-2],[1,0,-1]]))
    Gy = convolve(img, np.array([[1,2,1], [0,0,0],[-1,-2,-1]]))
    magnitude = np.sqrt(Gx**2 + Gy**2)
    plt.imsave('bilder/canny.png', magnitude, cmap='gray')
    return magnitude


def convolve(img, kernel):
    m,n = img.shape
    m_,n_ = kernel.shape
    y_range = [i for i in range(int(m_/2), m-int(m_/2))]
    x_range = [i for i in range(int(n_/2), n-int(n_/2))]
    result = np.zeros((m,n))


    for i in range(int(m_/2), m-int(m_/2)):
        for j in range(int(n_/2),n-int(n_/2)):
            s = 0
            for ii in range(0,m_):
                for jj in range(0,n_):
                    s += img[i-ii][j-jj]*kernel[ii][jj]
            result[i][j] = s
    return result
    
    #plt.imsave('bilder/blurred.png', trunc_img, cmap='gray')

    #Gauss filter er symmetrisk så vi trenger ikke rotere filterkjernen




def gaussian(size, sigma=1):
    x,y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


if __name__ == "__main__":
    main()
