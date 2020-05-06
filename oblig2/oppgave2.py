from scipy import signal
import time
import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
from numba import jit




class Kompresjon:
    def __init__(self, filnavn, q):
        self.filnavn = filnavn
        self.q = q
        self.Q = [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68,
                                                     109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]
        self.bilde = imread(filnavn, flatten=True)

    @jit
    def oppgave2(self):
        q = self.q
        Q = self.Q
        qQ = np.multiply(q, Q)
        bilde = self.bilde
        m,n = bilde.shape
        bilde -= 128
        
        m_ = int(m/8)
        n_ = int(n/8)
        blokker = np.zeros((m_,n_), dtype=np.ndarray)
        rekonstruerte_blokker = np.zeros((m_, n_), dtype=np.ndarray)

        for i in range(0,m,8):
            for j in range(0,n,8):
                blokker[int(i/8)][int(j/8)] = self.transform(bilde[i:i+8,j:j+8])
                blokker[int(i/8)][int(j/8)] = np.round(np.divide(blokker[int(i/8)][int(j/8)], qQ))
        
        """ print("Entropi for transformert bilde med kompresjonsrate q={0}: {1:.4g}".format(q, 
            self.beregn_entropi(self.utvid_blokker(blokker)))) """

        """ for i in range(m_):
            for j in range(n_):
                blokker[i][j] = np.round(np.divide(blokker[i][j], qQ)) """
        
        print("Entropi for kvantifisert bilde med kompresjonsrate q={0}: {1:.4g}".format(q, 
            self.beregn_entropi(self.utvid_blokker(blokker))))
        
        
        for i in range(m_):
            for j in range(n_):
                blokker[i][j] = np.round(np.multiply(blokker[i][j], qQ))
                rekonstruerte_blokker[i][j] = self.inv_transform(blokker[i][j])

        rekonstruerte_blokker += 128

        rekonstruert = self.utvid_blokker(rekonstruerte_blokker)
        
        plt.imsave('bilder/rekonstruert_uio_q={}.png'.format(q), rekonstruert, cmap='gray')

    @jit
    def transform(self, f):
        F = np.zeros((8, 8))
        for v in range(8):
            for u in range(8):
                left = (1/4)*self.c(u)*self.c(v)
                right = 0
                for y in range(8):
                    for x in range(8):
                        right += self.dct(f, x, y, u, v)
                F[u][v] = left*right
        return F


    def dct(self,f,x,y,u,v):
        return f[y][x]*math.cos(((2*y+1)*u*math.pi)/16)*math.cos(((2*x+1)*v*math.pi)/16)

    def c(self,a):
        if (a == 0):
            return 1/math.sqrt(2)
        else:
            return 1
    @jit
    def inv_transform(self,F):
        f = np.zeros((8,8))
        for y in range(8):
            for x in range(8):
                s = 0
                for v in range(8):
                    for u in range(8):
                        s += self.c(u)*self.c(v)*F[u][v]*math.cos((2*x+1)*v*math.pi/16)*math.cos((2*y+1)*u*math.pi/16)
                f[y][x] = round(s/4)
        return f
    
    def beregn_entropi(self, A):
        A = A.ravel()
        n  = len(A)
        hist = np.divide(np.histogram(A, bins=256)[0],n)
        res = 0
        for i in range(256):
            p_a = hist[i]
            if p_a > 1E-11:
                res += p_a*math.log(p_a,2)
        return -res


    @jit
    def utvid_blokker(self, blokker):
        m,n = blokker.shape
        m_,n_ = self.bilde.shape
        resultat = np.zeros((m_,n_))

        for i in range(0,m_,8):
            for j in range(0,n_,8):
                resultat[i:i+8,j:j+8] = blokker[int(i/8)][int(j/8)]
        return resultat



if __name__ == "__main__":
    q = [0.1,0.5,2,8,32]
    
    
    for e in q:
        k = Kompresjon('bilder/uio.png', e)
        k.oppgave2() 
