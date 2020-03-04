import numpy as np
from scipy.misc import imread, imshow
import matplotlib.pyplot as plt

image = imread("mona.png")
n, m = image.shape
d = np.zeros(255)
for i in range(n):
    for j in range(m):
        d[int(image[i][j])] += 1

plt.plot(d)
plt.show()

