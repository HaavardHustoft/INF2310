import math

D = 10E-3
f = 50E-3
s = 5
lambda_ = 500E-9

#a)
theta = (1.22*lambda_)/(D)
print("Theta: %f" % theta)
y = theta*s
print("Den minste avstanden er %f meter = %f mm" % (y, y*10**3))

#b)
y_ = y*f/(s-f)
print("Avstanden er %f mm i bildeplanet" % (y_*10**3))

#c)
T_0 = y_
f_0 = 1/T_0
print("Minste periode: T_0=%f\nHøyeste frekvens: f_0=%f" %(T_0,f_0))

#d)
