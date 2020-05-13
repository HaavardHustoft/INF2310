import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np
from scipy.signal import convolve2d

class EdgeDetector:
    def __init__(self):
        self.edgeFilter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    

    def read_image(self, filename):
        self.filename = filename
        self.img = self.img = imread('bilder/'+filename, flatten=True)
        self.n, self.m = self.img.shape


    def main(self, filename):
        #self.color_image(filename)
        self.read_image(filename)
        self.filter_imagedomain()
        self.filter_frequencydomain()
    

    def filter_imagedomain(self):
        n = self.n
        m = self.m
        img = self.img
        edgeFilter = self.edgeFilter
        Gx = convolve2d(img, np.array([[1,0,-1],[2,0,-2],[1,0,-1]]), mode='same', boundary='fill')
        Gy = convolve2d(img, np.array([[1,2,1],[0,0,0],[-1,-2,-1]]), mode='same', boundary='fill')
        res = np.sqrt(Gx**2 + Gy**2)
        normalized_res = np.divide(res,np.max(res))
        self.saveImage(res, 'bilder/sobel_'+self.filename)
        self.saveImage(img+res, 'bilder/sobel_bedre'+self.filename)
    

    def filter_frequencydomain(self):
        n = self.n
        m = self.m
        expanded_filter = np.zeros((n,m))
        expanded_filter[:self.edgeFilter.shape[0] , :self.edgeFilter.shape[1]] = self.edgeFilter

        filter_fourier = np.fft.fft2(expanded_filter)
        img_fourier = np.fft.fft2(self.img)
        result = np.real(np.fft.ifft2(img_fourier*filter_fourier))
        self.saveImage(result, 'bilder/fourierkanter_'+self.filename)

    def improve_image(self, img, res):
        newImg = img + res
        return newImg
    

    def saveImage(self, newImg, filename):
        plt.imsave(filename, newImg, cmap='gray')
    

    def color_image(self, filename):
        img = imread('bilder/'+filename, mode='RGB')
        gray_img = imread('bilder/'+filename, flatten=True)
        m,n = gray_img.shape
        edgeFilter = self.edgeFilter
        Gx = convolve2d(gray_img, np.array(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]]), mode='same', boundary='fill')
        Gy = convolve2d(gray_img, np.array(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), mode='same', boundary='fill')
        res = np.sqrt(Gx**2 + Gy**2)
        result = np.zeros((gray_img.shape))
        for i in range(m):
            for j in range(n):
                val = result[i][j]
                result[i][j] = round((val*img[i][j][0] + val*img[i][j][1] + val*img[i][j][2])/3)
        self.saveImage(result, 'bilder/sobel_'+filename)

        



if __name__ == "__main__":
    edgeDetector = EdgeDetector()
    edgeDetector.main(input('filnavn: ')+'.png')


