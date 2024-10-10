import numpy as np

e = 0.1
rx = 0.5
ry = 0.5
m = 21
n = 1000000
alpha_2 = 0.03

def get_lxly_nn_grr_error(lx, ly):
    error = ((2 * alpha_2 * (lx*rx+ly*ry))/(lx*ly))**2 + (lx * rx * ly * ry * m * (np.exp(e) + lx * ly - 2)) / n*((np.exp(e) - 1)**2)
    return error


def get_lxly_nn_oue_error(lx, ly):
    error = ((2*alpha_2*(lx*rx + ly*ry))/(lx*ly))**2 + (4*lx*rx*ly*ry*m*np.exp(e)) / n*((np.exp(e) - 1)**2)
    return error

def get_lxly_cn_grr_error(lx, b, ry):
    error = ((2*alpha_2*ry)/lx)**2 + (lx*rx*b*ry*m(np.exp(e) + lx*b - 2)) / n*((np.exp(e) - 1)**2)
    return error


def get_lxly_cn_oue_error(lx, b, ry):
    error = ((2*alpha_2*ry)/lx)**2 + (4*lx*rx*b*ry*m*np.exp(e)) / n*((np.exp(e) - 1)**2)
    return error

def variance_oue(ep):
    return (4*np.exp(ep)) / n*((np.exp(ep) - 1)**2)

def variance_grr(ep):
    return (np.exp(ep) + 4 - 2) / n*((np.exp(ep) - 1)**2)

# for i in range(1, 5, 1):
#     print("{:.2f} {:.10f}".format(i, variance_grr(i)))

for i in range(1, 50, 1):
    print("{:.2f} {:.10f}".format(i, get_lxly_nn_grr_error(i, 2)))