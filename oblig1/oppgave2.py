import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import math
import sys
from skimage import exposure
from skimage import data, img_as_float

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
    #huang('bilder/oppgave2/oppg2_skalert.png')

    #compare(filename)
    #clahe(out_image, m, n, s)
    huang('bilder/oppgave2/oppg2_skalert.png')

    


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
    cutoff = 0.5


    segment = np.ndarray((s,s))
    
    # sliding window ahe
    for i in range(s):
        for j in range(s):
            segment[i][j] = image[i][j]
    c1 = find_cumulative_histogram_clahe(segment)
    T1 = [(G-1)*c1[k] for k in range(G)]
    image[1][1] = T1[int(image[1][1])]

    for i in range(s,m-1):
        for j in range(s, n-2):
            for k in range(s):
                segment[k][0] = segment[k][1]
                segment[k][1] = segment[k][2]
                segment[k][2] = image[i][j+2]
            c = find_cumulative_histogram_clahe(segment)
            T = [(G-1)*c[k] for k in range(G)]
            image[i][j] = T[int(image[i][j])]
    save(image, 'bilder/oppgave2/clahe_result_cutoff={}.png'.format(cutoff), 'bilder/oppgave2/clahe_hist_cutoff={}.png'.format(cutoff),cutoff)
    

def find_cumulative_histogram_clahe(segment):
    m,n = segment.shape
    N = m*n
    histogram = np.zeros(256)

    for i in range(n):
        for j in range(m):
            histogram[int(segment[i][j])] += 1
    print(histogram)
    normal_histogram = histogram/N
    cutoff = np.max(normal_histogram)
    
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



def huang(filename):
    image = imread(filename,flatten=True)
    m,n = image.shape
    N = n*m
    w = 3 # window size
    
    
    for i in range(1,m-1):
        #Finner første histogrammet for hver rad
        hist = np.zeros(256)
        for v in range(i-1, i+2):
            for g in range(3):
                hist[int(image[v][g])] += 1
        cdh = find_cdh(hist, w)
        pi = int(image[i][1])
        p = cdh[pi]*255
        
        #Denne løkken trekker fra siste kolonne og legger til den nye kolonnen til høyre
        for k in range(-1,1):
            hist[int(image[i+k][0])] -= 1
            hist[int(image[i+k][3])] += 1
        
        if (p > 255):
            image[i][1] = 255
        else:
            image[i][1] = p
        #legger til den nye pikselen i histogrammet
        hist[int(image[i][1])] += 1
        #fjerner den gamle verdien siden den ikke finnes i bildet lenger
        hist[pi] -= 1
        
        for j in range(2,n-2):
            cdh = find_cdh(hist, w)
            pi = int(image[i][j])
            p = cdh[pi]*255
            

            for k in range(-1,1):
                hist[int(image[i+k][j-1])] -= 1
                hist[int(image[i+k][j+2])] += 1
            
            if (p > 255):
                image[i][j] = 255
            else:
                image[i][j] = p
            
            hist[int(image[i][j])] += 1
            hist[pi] -= 1
        cdh = find_cdh(hist,w)
        p = cdh[int(image[i][n-2])]
        old = [int(image[i][n-2])]
        if (p > 255):
            image[i][j] = 255
        else:
            image[i][j] = p

    save(image, 'bilder/oppgave2/huang.png', 'bilder/oppgave2/huang_hist.png')
                
    

def find_cdh(histogram, w):
    normal_histogram = histogram/(w*w)
    
    lim = 0.05
    acc = 0
    for i in range(256):
        if (normal_histogram[i] > lim):
            diff = normal_histogram[i]-lim
            normal_histogram[i] = normal_histogram[i]-diff
            acc = acc + diff
    normal_histogram += acc/256
    
    cdh = np.zeros(256)
    cdh[0] = normal_histogram[0]
    for i in range(1,256):
        cdh[i] = cdh[i-1] + normal_histogram[i]
    return cdh

    

    

def compare(filename):
    img = imread(filename)
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 4), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5+i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title('Contrast stretching')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title('Histogram equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title('Adaptive equalization')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()



def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def save(image, name, hist_name,cutoff=0):
    fig = plt.figure()
    plt.hist(image.ravel(), bins=256)
    plt.title('Histogram for clahe med cutoff={}'.format(cutoff))
    plt.savefig(hist_name)
    plt.close(fig)
    plt.imsave(name, image, cmap='gray')




if __name__ == "__main__":
    main(sys.argv[1])