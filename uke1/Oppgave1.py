from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np

filename = 'mona.png'
f = imread(filename, flatten=True)


N, M = f.shape
f_out = np.zeros((N,M))
for i in range(1,N):
    for j in range(M):
        f_out[i,j] = f[i,j]-f[i-1,j]

bias = 128
plt.figure()
plt.imshow(f_out*1.5+128,cmap='gray',vmin=0,vmax=255,aspect='auto')
plt.title('f_out with bias and contrast')

plt.figure()
plt.imshow(f_out*1.5,cmap='gray',vmin=0,vmax=255,aspect='auto')
plt.title('f_out with added contrast')

plt.figure()
plt.imshow(f_out + 128,cmap='gray',vmin=0,vmax=255,aspect='auto')
plt.title('f_out with added bias')

plt.figure()
plt.imshow(f_out,cmap='gray',vmin=0,vmax=255,aspect='auto')
plt.title('f_out')

plt.figure()
plt.imshow(f, cmap='gray')
plt.title('Original')


plt.show()