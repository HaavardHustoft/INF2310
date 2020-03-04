#1)
D = 0.02
lambda_ = 410E-9
S = 400000000
theta = 1.22*lambda_/D
y = theta*S
print("Den minste avtsanden mellom to gjenkjennelige punkter på månen er %.0f meter, eller %.0f km" % (y,y/1000))

#2)
D_2 = 0.5
S_2 = 150000000000
lambda_2 = 410E-9
theta_2 = 1.22*lambda_2/D_2
y_2 = theta_2*S_2
print("Den minste avstanden mellom to gjenkjennelige punkter på sola er %.0f meter, eller %.0f km" % (y_2, y_2/1000))
