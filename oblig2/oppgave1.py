from scipy import signal
import time
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt


img = imread("bilder/cow.png", flatten=True)

middelverdifilter = np.ndarray((15,15))
total = 15*15
for i in range(15):
    for j in range(15):
        middelverdifilter[i][j] = 1/total

img_1 = signal.convolve2d(img, middelverdifilter)
plt.imsave("bilder/middelverdifiltrert_15.png", img_1, cmap="gray")


img_2 = np.fft.fftn(np.fft.fft2(img)*np.fft.fft2(middelverdifilter))
plt.imsave("bilder/fourier_middelverdi.png",img_2 , cmap="gray")

