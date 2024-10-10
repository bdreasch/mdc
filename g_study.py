import numpy as np

n = 1000000
m = 10
ep = 1
alpha = 0.7
r = 2

g = ((n * (np.exp(ep - 1))**2 * r * alpha**2) / (4 * m * np.exp(ep)))**(1/3)

print(g)