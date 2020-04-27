import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import math
import sys
from skimage import exposure
from skimage import data, img_as_float

def main(filename):
    image = imread(filename, flatten=True)
    m,n = image.shape
    window_size = int(m/8)
    #global_utjevning(image, m, n)
    m_ = m
    n_ = n
    if (m % window_size != 0):
        m_ = m + int(window_size/2)
    if (n%window_size != 0):
        n_ = n + int(window_size)
    
    out_image = np.ndarray((m_,n_))

    plt.figure()
    plt.hist(image.ravel(), bins=256)
    plt.savefig('bilder/original_hist.png')

    #skalerer bildet
    for y in range(m_):
        for x in range(n_):
            src_y = min(int(round(float(y)/float(m_)*float(m))), m-1)
            src_x = min(int(round(float(x)/float(n_)*float(n))), n-1)
            src = image[src_y][src_x]
            out_image[y][x] = src
    plt.imsave('bilder/oppgave2/oppg2_skalert.png', out_image, cmap='gray')

    global_utjevning('bilder/oppgave2/oppg2_skalert.png') # kan legge til ekstra argument for bits_ut her dersom det er ønskelig, standard er bits_ut=8
    ahe('bilder/oppgave2/oppg2_skalert.png',window_size)
    huang('bilder/oppgave2/oppg2_skalert.png', window_size)

    


def global_utjevning(filename, bits_ut=8):
    image = imread(filename, flatten=True)
    m,n = image.shape
    N = m*n

    histogram = np.zeros(256)
    for i in range(len(image.ravel())):
        histogram[int(round(image.ravel()[i]))] += 1
    
    normal_histogram = histogram/N
    cumulative = np.zeros(256)
    cumulative[0] = normal_histogram[0]

    for i in range(1, len(normal_histogram)):
        cumulative[i] = cumulative[i-1] + normal_histogram[i]
    G = 2**bits_ut
    T = [round((G-1)*i) for i in cumulative]

    out_image = np.ndarray((m,n))
    for i in range(m):
        for j in range(n):
            out_image[i][j] = T[int(image[i][j])]
    
    plt.imsave('bilder/oppgave2/oppg2_global.png', out_image, cmap='gray')
    fig = plt.figure()
    plt.hist(out_image.ravel(), bins=256)
    plt.title('Utjevnet histogram')
    plt.savefig('bilder/oppgave2/oppg2_global_hist.png')
    plt.close(fig)
    


def ahe(filename, window_size):
    image = imread(filename,flatten=True)
    m,n = image.shape
    N = n*m
    half = int(window_size/2)
    
    
    for i in range(half,m-half):
        #Finner første histogrammet for hver rad
        hist = np.zeros(256)
        for v in range(i-half, i+half+1):
            for g in range(window_size):
                hist[int(image[v][g])] += 1
        cdh = find_cdh_ahe(hist, window_size)
        pi = int(image[i][half])
        p = cdh[pi]*255
        
        #Denne løkken trekker fra siste kolonne og legger til den nye kolonnen til høyre
        for k in range(-half,half):
            hist[int(image[i+k][0])] -= 1
            hist[int(image[i+k][window_size])] += 1
        
        if (p > 255):
            image[i][half] = 255
        else:
            image[i][half] = p
        #legger til den nye pikselen i histogrammet
        hist[int(image[i][half])] += 1
        #fjerner den gamle verdien da den ikke finnes i bildet lenger
        hist[pi] -= 1
        
        for j in range(half+1,n-(half+1)):
            cdh = find_cdh_ahe(hist, window_size)
            pi = int(image[i][j])
            p = cdh[pi]*255

            for k in range(-half,half):
                hist[int(image[i+k][j-half])] -= 1
                hist[int(image[i+k][j+half+1])] += 1
            if (p > 255):
                image[i][j] = 255
            else:
                image[i][j] = p
            
            hist[int(image[i][j])] += 1
            hist[pi] -= 1
        cdh = find_cdh_ahe(hist,window_size)
        p = cdh[int(image[i][n-2])]
        old = [int(image[i][n-2])]
        if (p > 255):
            image[i][j] = 255
        else:
            image[i][j] = p

    save(image,'bilder/oppgave2/ahe.png','bilder/oppgave2/ahe_hist.png')
            

def find_cdh_ahe(histogram, w):
    normal_histogram = histogram/(w*w)
    
    cdh = np.zeros(256)
    cdh[0] = normal_histogram[0]
    for i in range(1,256):
        cdh[i] = cdh[i-1] + normal_histogram[i]
    return cdh



def huang(filename, window_size):
    image = imread(filename,flatten=True)
    m,n = image.shape
    N = n*m
    half = int(window_size/2)
    
    
    for i in range(half,m-half):
        #Finner første histogrammet for hver rad
        hist = np.zeros(256)
        for v in range(i-half, i+half+1):
            for g in range(window_size):
                hist[int(image[v][g])] += 1
        cdh = find_cdh_clahe(hist, window_size)
        pi = int(image[i][half])
        p = cdh[pi]*255
        
        #Denne løkken trekker fra siste kolonne og legger til den nye kolonnen til høyre
        for k in range(-half,half):
            hist[int(image[i+k][0])] -= 1
            hist[int(image[i+k][window_size])] += 1
        
        if (p > 255):
            image[i][half] = 255
        else:
            image[i][half] = p
        #legger til den nye pikselen i histogrammet
        hist[int(image[i][half])] += 1
        #fjerner den gamle verdien da den ikke finnes i bildet lenger
        hist[pi] -= 1
        
        for j in range(half+1,n-(half+1)):
            cdh = find_cdh_clahe(hist, window_size)
            pi = int(image[i][j])
            p = cdh[pi]*255

            for k in range(-half,half):
                hist[int(image[i+k][j-half])] -= 1
                hist[int(image[i+k][j+half+1])] += 1
            if (p > 255):
                image[i][j] = 255
            else:
                image[i][j] = p
            
            hist[int(image[i][j])] += 1
            hist[pi] -= 1
        cdh = find_cdh_clahe(hist,window_size)
        p = cdh[int(image[i][n-2])]
        old = [int(image[i][n-2])]
        if (p > 255):
            image[i][j] = 255
        else:
            image[i][j] = p

    save(image, 'bilder/oppgave2/huang_window_size={}.png'.format(window_size), 'bilder/oppgave2/huang_hist_window_size={}.png'.format(window_size))
    

def find_cdh_clahe(histogram, w):
    normal_histogram = histogram/(w*w)
    lim = 0.025
    acc = 0
    for i in range(256):
        if (normal_histogram[i] > lim):
            diff = normal_histogram[i]-lim
            normal_histogram[i] -= diff
            acc += diff
    normal_histogram += acc/256
    
    cdh = np.zeros(256)
    cdh[0] = normal_histogram[0]
    for i in range(1,256):
        cdh[i] = cdh[i-1] + normal_histogram[i]
    return cdh


def save(image, name, hist_name):
    fig = plt.figure()
    plt.hist(image.ravel(), bins=256, density=True)
    plt.savefig(hist_name)
    plt.close(fig)
    plt.imsave(name, image, cmap='gray')


if __name__ == "__main__":
    main(sys.argv[1])