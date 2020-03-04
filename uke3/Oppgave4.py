import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import math
import argparse



def main():
    parser = argparse.ArgumentParser(description='Gray scale transformations')
    parser.add_argument('Transform', metavar='--T', nargs='+', help='Type of transform to use')
    parser.add_argument('Image', metavar='--I', nargs='+', help='Filename of image to work with')
    args = parser.parse_args()

    image_data = imread(args.Image[0], flatten=True)
    m, n = image_data.shape
    alt_data = image_data.copy()


    if (args.Transform[0] == 'Linear'):
        sigma = np.std(image_data.ravel())
        my = np.mean(image_data.ravel())
        print("Original \u03C3 : %.2f \nOriginal \u03BC : %.2f" % (sigma,my))

        sigma_t = float(input("New sigma:"))
        my_t = float(input("New my:"))
        a = sigma_t/sigma
        b = my_t - my*a

        print("\u03C3 = %.2f and \u03BC = %.2f gives a = %.2f and b = %.2f" %(sigma_t,my_t,a,b))

        for i in range(0,n):
            for j in range(0,m):
                alt_data[i][j] = linear_transform(image_data[i][j], a, b)
        plot(image_data, alt_data)
    

    elif (args.Transform[0] == 'exp'):
        c = float(input('c?'))
        gamma = float(input('Gamma?'))

        for i in range(0,n):
            for j in range(0,m):
                alt_data[i][j] = exp_transform(image_data[i][j], c, gamma)
        plot(image_data, alt_data)
    
    elif (args.Transform[0] == 'log'):
        c = float(input('c?'))
        for i in range(0,n):
            for j in range(0,m):
                alt_data[i][j] = log_transform(image_data[i][j], c)
        plot(image_data, alt_data)


def linear_transform(r, a, b):
    new_r = a*r + b
    if (new_r <= 255):
        return new_r
    else:
        return 255


def exp_transform(r, c, gamma):
    new_r = c*r**gamma
    if (new_r <= 255):
        return new_r
    else:
        return 255

def log_transform(r, c):
    new_r = c*math.log(r + 1)

    if (new_r <= 255):
        return new_r
    else:
        return 255


def plot(img_1, img_2):
    img = np.concatenate((img_1,img_2),axis=1)
    plt.figure()
    plt.imshow(img, cmap='gray')

    plt.figure()
    plt.hist([img_1.ravel(),img_2.ravel()], bins=256, color=["red","green"],label=["Original","Transform"])
    plt.legend(loc='upper right')
    plt.show()



if __name__ == "__main__":
    main()