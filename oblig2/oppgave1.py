from scipy import signal
import time
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def lag_filter(dim):
    middelverdifilter = np.ndarray((dim,dim))
    total = dim*dim
    for i in range(dim):
        for j in range(dim):
            middelverdifilter[i][j] = 1/total
    return middelverdifilter


def oppgave_11(img):
    middelverdifilter = lag_filter(15)

    img_1 = signal.convolve2d(img, middelverdifilter, 'same')
    plt.imsave("bilder/middelverdifiltrert_15.png", img_1, cmap="gray")


    utvidet_filter = np.zeros(img.shape)
    utvidet_filter[:middelverdifilter.shape[0], :middelverdifilter.shape[1]] = middelverdifilter

    filter_fourier = np.fft.fft2(middelverdifilter, s=(img.shape[1],img.shape[0]))
    image_fourier = np.fft.fft2(img)
    img_2 = np.fft.ifft2(image_fourier*filter_fourier)
    img_2 = np.real(img_2)
    plt.imsave("bilder/fourier_middelverdi.png",img_2 , cmap="gray")



def oppgave_13(img, filter_dim):
    
    middelverdifilter = lag_filter(filter_dim)

    start = timer()
    filter_fourier = np.fft.fft2(middelverdifilter, s=(img.shape[1], img.shape[0]))
    image_fourier = np.fft.fft2(img)
    img_2 = np.fft.ifft2(image_fourier*filter_fourier)
    img_2 = np.real(img_2)
    fourier = timer() - start


    start = timer()
    img_1 = signal.convolve2d(img, middelverdifilter, 'same')
    romlig = timer() - start

    return (romlig,fourier)

def plot(romlig,fourier, n):
    plt.figure()
    plt.plot(n, romlig, 'g', label='Romlig')
    plt.plot(n, fourier, 'r', label='Fourier')
    plt.xlabel('Filterstørrelse')
    plt.ylabel('Sekunder')
    plt.title('Kjøretider målt i sekunder')
    plt.legend()
    plt.savefig('bilder/oppgave1_3.png')


def tid(img):
    tider_fourier = []
    tider_romlig = []
    n = [i for i in range(5,100,5)]

    for i in n:
        tider_romlig.append(oppgave_13(img, i)[0])
        tider_fourier.append(oppgave_13(img, i)[1])
    plot(tider_romlig, tider_fourier, n)

if __name__ == "__main__":
    img = imread("bilder/cow.png", flatten=True)
    oppgave_11(img)
    
