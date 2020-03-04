import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt


kernel = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]))
image = imread('cellekjerner.png',flatten=True)
convolve(image, kernel)



def convolve(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    i = 0
    j = 0
    k = kernel_row-1
    l = kernel_col-1
"""
    while (i < image_row & k >= 0):
        while (j < image_col-1 & l >= 0):
            

     for i in range(image_row):
        for j in range(image_col):
            accumulator = 0
            
            for k in range(kernel_row):
                for l in range(kernel_col): """

