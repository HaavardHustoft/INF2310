import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import math
import sys

def main(filename, bits_ut=8):
    image = imread(filename, flatten=True)
    m,n = image.shape

    #global_utjevning(image, m, n)

    s = 3 # sidelengden på kvadratet
    new_height = m + s%(m+s)
    new_width = n + s%(n+s)
    out_image = np.ndarray((new_height,new_width))

    plt.figure()
    plt.hist(image.ravel(), bins=256)
    plt.savefig('bilder/original_hist.png')

    #skalerer bildet slik at vi ikke mister piksler ved bilderanden
    for y in range(new_height):
        for x in range(new_width):
            src_y = min(int(round(float(y)/float(new_height)*float(m))), m-1)
            src_x = min(int(round(float(x)/float(new_width)*float(n))), n-1)
            src = image[src_y][src_x]
            out_image[y][x] = src
    plt.imsave('bilder/oppgave2/oppg2_skalert.png', out_image, cmap='gray')

    clahe(out_image, m, n, s)
    


def global_utjevning(image,m,n):
    N = m*n

    histogram = np.zeros(256)
    for i in range(len(image.ravel())):
        histogram[int(round(image.ravel()[i]))] += 1
    
    normal_histogram = histogram/N
    cumulative = np.zeros(256)
    cumulative[0] = normal_histogram[0]

    for i in range(1, len(normal_histogram)):
        cumulative[i] = cumulative[i-1] + normal_histogram[i]
    
    T = [round((256-1)*i) for i in cumulative]

    out_image = np.ndarray((m,n))
    for i in range(m):
        for j in range(n):
            out_image[i][j] = T[int(image[i][j])]
    
    plt.imsave('bilder/oppgave2/oppg2_utjevning.png', out_image, cmap='gray')
    fig = plt.figure()
    plt.hist(out_image.ravel(), bins=256)
    plt.title('Utjevnet histogram')
    plt.savefig('bilder/oppgave2/oppg2_hist.png')
    plt.close(fig)
    


def ahe(image, m, n, s):
    N = m*n
    G = 256


    segment = np.ndarray((s,s))
    # sliding window ahe
    for i in range(s):
        for j in range(s):
            segment[i][j] = image[i+1][j+1]
    c = find_cumulative_histogram(segment)
    T = [(G-1)*c[k] for k in range(G)]
    image[2][2] = T[int(image[2][2])]

    for i in range(1,m-1):
        for j in range(1, n-2):
            for k in range(s):
                segment[k][0] = segment[k][1]
                segment[k][1] = segment[k][2]
                segment[k][2] = image[i][j+2]
            c = find_cumulative_histogram(segment)
            T = [(G-1)*c[k] for k in range(G)]
            image[i][j] = T[int(image[i][j])]


    """ for i in range(1, new_height-1):
        for j in range(1,new_width-1):
            segment = np.ndarray((s,s))

            for v in range(s):
                segment[v] = out_image[i-(v-1)][j-1:j+s-1]
            
            c = find_cumulative_histogram(segment)
            T = [(G-1)*c[k] for k in range(G)]
            out_image[i][j] = T[int(out_image[i][j])] """
    save(image)
            

def find_cumulative_histogram(segment):
    m,n = segment.shape
    N = m*n
    normal_histogram = np.zeros(256)

    for i in range(n):
        for j in range(m):
            normal_histogram[int(segment[i][j])] += 1
    
    normal_histogram = normal_histogram/N
    normal_cumulative_histogram = np.zeros(256)
    normal_cumulative_histogram[0] = normal_histogram[0]


    for i in range(1,len(normal_histogram)):
        normal_cumulative_histogram[i] = normal_cumulative_histogram[i-1] + normal_histogram[i]
    
    return normal_cumulative_histogram


def clahe(image, m, n, s):
    N = m*n
    G = 256
    cutoff = 0.08


    segment = np.ndarray((s,s))
    
    # sliding window ahe
    for i in range(s):
        for j in range(s):
            segment[i][j] = image[i][j]
    c1 = find_cumulative_histogram_clahe(segment, cutoff)
    T1 = [(G-1)*c1[k] for k in range(G)]
    image[1][1] = T1[int(image[1][1])]

    for i in range(s,m-1):
        for j in range(s, n-2):
            for k in range(s):
                segment[k][0] = segment[k][1]
                segment[k][1] = segment[k][2]
                segment[k][2] = image[i][j+2]
            c = find_cumulative_histogram_clahe(segment, cutoff)
            T = [(G-1)*c[k] for k in range(G)]
            image[i][j] = T[int(image[i][j])]
    save(image, 'bilder/oppgave2/clahe_result_cutoff={}.png'.format(cutoff), 'bilder/oppgave2/clahe_hist_cutoff={}.png'.format(cutoff),cutoff)
    

def find_cumulative_histogram_clahe(segment, cutoff):
    m,n = segment.shape
    N = m*n
    normal_histogram = np.zeros(256)

    for i in range(n):
        for j in range(m):
            normal_histogram[int(segment[i][j])] += 1
    
    normal_histogram /= N
    accumulator = 0
    for i in range(256):
        if (normal_histogram[i] > cutoff):
            diff = normal_histogram[i] - cutoff
            normal_histogram[i] -= diff
            accumulator += diff
    normal_histogram += accumulator/256
    normal_cumulative_histogram = np.zeros(256)
    normal_cumulative_histogram[0] = normal_histogram[0]


    for i in range(1,len(normal_histogram)):
        normal_cumulative_histogram[i] = normal_cumulative_histogram[i-1] + normal_histogram[i]
    return normal_cumulative_histogram

def save(image, name, hist_name,cutoff=0):
    fig = plt.figure()
    plt.hist(image.ravel(), bins=256)
    plt.title('Histogram for clahe med cutoff={}'.format(cutoff))
    plt.savefig(hist_name)
    plt.close(fig)
    plt.imsave(name, image, cmap='gray')




if __name__ == "__main__":
    main(sys.argv[1])