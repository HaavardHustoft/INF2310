import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import math

image = misc.imread(name='car.png', flatten=True)


n,m = image.shape
N = n*m

hist = np.histogram(image, bins=256)
normal_hist = hist[0]/N
cum_hist = np.zeros(256)
cum_hist[0] = normal_hist[0]
T = np.zeros(256)

for i in range(1,256):
    cum_hist[i] = cum_hist[i-1] + normal_hist[i]

T = [math.floor((255)*cum_hist[i]) for i in range(256)]

alt_image = image.copy()

for i in range(n):
    for j in range(m):
        alt_image[i][j] = T[int(round(image[i][j]))]


alt_hist = np.histogram(alt_image, bins=256)
alt_cum_hist = np.zeros(256)
alt_cum_hist[0] = alt_hist[0][0]
for i in range(1,256):
    alt_cum_hist[i] = alt_cum_hist[i-1] + alt_hist[0][i]


plt.figure()
plt.plot(alt_cum_hist)

img = np.concatenate((image, alt_image), axis=1)



plt.figure()
plt.imshow(img, cmap='gray')
plt.show()






""" def find_sum(cum_arr, arr, i):
    s = cum_arr[i-1] + arr[i]
    return s


for i in range(1,N):
    cumulative_hist[i] = find_sum(cumulative_hist,image.ravel(), i)

histogram = np.histogram(image)
s = 0
normal = histogram[0]/N

plt.figure()
plt.plot(cumulative_hist)


plt.figure()
plt.plot(normal)

plt.figure()
plt.hist(image.ravel(), bins=256)
plt.show() """
 