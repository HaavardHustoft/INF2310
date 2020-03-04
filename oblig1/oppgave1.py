import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import math

def main():
    image = imread('bilder/portrett.png', flatten=True)
    
    """ plt.hist(image.ravel(), bins=256)
    plt.title('Histogram av portrett.png')
    plt.savefig('bilder/portrett_histogram.png')

    std = 64
    my = 127

    new_image = linear_transform(image.copy(), std, my)

    plt.figure()
    plt.hist(new_image.ravel(),bins=256)
    plt.title('Linear transform histogram,std=%.2f,mean=%.2f' % (np.std(new_image),np.mean(new_image)))
    plt.savefig('bilder/linear_transform_histogram.png')

    plt.figure()
    plt.imshow(new_image, cmap='gray')
    plt.title('Linear transform')
    plt.savefig('bilder/linear_transform_portrett.png') """

    geometric_transform(image)




def linear_transform(image, std, my):
    n,m = image.shape
    old_std = np.std(image.ravel())
    old_my = np.mean(image.ravel())

    a = std/old_std
    b = my - old_my*a

    for i in range(n):
        for j in range(m):
            image[i][j] = min(a*image[i][j] + b, 255)
    return image



def geometric_transform(image):
    out_image = imread('bilder/geometrimaske.png', flatten=True)
    out_image_2 = out_image.copy()
    width, height = image.shape
    new_width,new_height = out_image.shape

    plt.figure()
    plt.imshow(out_image, cmap='gray')
    plt.show()

    #forlengs

    #baklengs
    for x in range(new_width):
        for y in range(new_height):
            src_x = min(int(round(float(x)/float(new_width)*float(width))), width-1)
            src_y = min(int(round(float(y)/float(new_height)*float(height))), height-1)
            src = image[src_x][src_y]
            out_image[x][y] = src
    
    plt.figure()
    plt.imshow(out_image,cmap='gray')
    plt.savefig('bilder/nytt_portrett_baklengs.png')



       
                



if __name__ == "__main__":
    main()