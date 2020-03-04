from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np

f = imread('mona.png',flatten=True)

bit_1 = 8
bit_2 = 6
bit_3 = 4
bit_4 = 2


noiseFactor = 20
N,M = f.shape
fNoisy = f + noiseFactor*np.random.randn(N,M)


plt.figure()
plt.imshow(fNoisy//2**(8-bit_4),cmap='gray',vmin=0,vmax=2**bit_4)
plt.title('Noisy')


plt.figure()
plt.imshow(f//2**(8-bit_4),cmap='gray',vmin=0,vmax=2**bit_4)
plt.title('2 bits')

""" plt.figure()
plt.imshow(f//2**(8-bit_3),cmap='gray',vmin=0,vmax=2**bit_3)
plt.title('4 bits')

plt.figure()
plt.imshow(f//2**(8-bit_2),cmap='gray',vmin=0,vmax=2**bit_2)
plt.title('6 bits')

plt.figure()
plt.imshow(f//2**(8-bit_1),cmap='gray',vmin=0,vmax=2**bit_1)
plt.title('8 bits') """

plt.show()


#Ser ut til at man kun kan gå ned til 6 bits før kvaliteten forringes betydelig

