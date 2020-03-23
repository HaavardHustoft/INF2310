import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import math

def main():
    original_image = imread('bilder/portrett.png', flatten=True)
    
    linear_transform(original_image)

    image = imread('bilder/oppgave1/linear_transform_portrett.png',flatten=True)

    maske = imread('bilder/geometrimaske.png')
    
    backward(image, maske)
    forward(image,maske)

    plt.figure()
    plt.hist(image.ravel(), bins=256)
    plt.title('Histogram av portrett.png')
    plt.savefig('bilder/portrett_histogram.png')
    


def forward(image,maske):
    m,n,k = maske.shape
    m_,n_ = image.shape

    new_image = np.ndarray((m,n))
    for i in range(m_):
        for j in range(n_):
            x_,y_ = T(j,i)
            y_rounded = int(y_)
            x_rounded = int(x_)
            if (x_rounded in range(0,n) and y_rounded in range(0,m)):
                new_image[y_rounded][x_rounded] = image[i][j]
    
    plt.imsave('bilder/oppgave1/nytt_portrett_forlengs.png',new_image,cmap='gray')

def backward(image,maske):
    m,n,k = maske.shape
    m_,n_ = image.shape
    new_image_1 = np.ndarray((m,n))
    new_image_2 = np.ndarray((m,n))

    for i in range(m):
        for j in range(n):
            x_,y_ = T_inverse(j,i)
            y_rounded = int(y_)
            x_rounded = int(x_)
            if (x_rounded in range(0,n_) and y_rounded in range(0,m_)):
                new_image_1[i][j] = bilinear_interpolate(image, x_, y_) #bilineær
                new_image_2[i][j] = image[y_rounded][x_rounded] # nærmeste nabo

    plt.imsave('bilder/oppgave1/nytt_portrett_baklengs_bilinear.png',new_image_1,cmap='gray')
    plt.imsave('bilder/oppgave1/nytt_portrett_baklengs_nabo.png',new_image_2,cmap='gray')

def T(x,y):
    x_ = 3.2*x - 3*y + 164.2
    y_ = 2.338658147*x + 4.092651757*y - 299.600639
    return x_,y_

def T_inverse(x,y):
    x_ = 0.20348837208868717251*x + 0.14916126572995922689*y + 11.276019829782152097
    y_ = -0.11627906977206701599*x + 0.15910535011195650869*y + 66.761087818434295571
    return x_,y_

def bilinear_interpolate(in_image, x, y):
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = math.ceil(x)
    y1 = math.ceil(y)

    delta_x = x-x0
    delta_y = y-y0

    p = in_image[y0][x0] + (in_image[y1][x0] - in_image[y0][x0])*delta_y
    q = in_image[y0][x1] + (in_image[y1][x1] - in_image[y0][x1])*delta_y

    new_val = p + (q - p)*delta_x
    return new_val


def linear_transform(image):
    std = 64
    my = 127

    new_image = image.copy()

    n,m = image.shape
    old_std = np.std(image.ravel())
    old_my = np.mean(image.ravel())

    a = std/old_std
    b = my - old_my*a

    for i in range(n):
        for j in range(m):
            new_image[i][j] = min(a*new_image[i][j] + b, 255)
    
    plt.imsave('bilder/oppgave1/linear_transform_portrett.png',new_image, cmap='gray')
    
    plt.hist(new_image.ravel(),bins=256)
    plt.title('Linear transform histogram,std=%.2f,mean=%.2f' % (np.std(new_image.ravel()),np.mean(new_image.ravel())))
    plt.savefig('bilder/oppgave1/linear_transform_histogram.png')

    


if __name__ == "__main__":
    main()