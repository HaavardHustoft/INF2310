import numpy as np
from math import sin,cos,pi,radians
from scipy.misc import imread

theta = radians(33)

def rotate(x,y):
    T = np.array([[cos(theta), -sin(theta), 0], [sin(theta),cos(theta),0], [0,0,1]])
    v = np.array([x,y,1])
    return T.dot(v)

print(rotate(207,421))

