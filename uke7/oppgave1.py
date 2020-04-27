import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt


kernel = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]))
image = imread('cellekjerner.png',flatten=True)
convolve(image, kernel)



def convolve(image, kernel): #Antar at kernel er symmetrisk
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    for i in range(image_row):
        for j in range(image_col):
            accumulator = image[i][j]*kernel[kernel_row-1][kernel_col-1]
            
            for k in range(kernel_row):
                for l in range(kernel_col):
                    


