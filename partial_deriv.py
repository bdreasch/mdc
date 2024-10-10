from sympy import symbols, cos, diff, exp, solve, Eq, nsolve, linsolve, I
import numpy as np

# a, n, m, e, r, g = symbols('a n m e r g', real=True)
# f = (a/g)**2 + (g/r)*(m*((g-2+exp(e))))/(n*(exp(e)-1)**2)

p=0.03
r=0.5
s=0.5
n=1000000
m=21


eps = [0.1, 0.5, 1, 1.5, 2.0, 2.5, 3]
for e in eps:
    a, b = symbols('a b', real=True, positive=True)
    f_grr = ((2*p*(a*r+b*s))/(a*b))**2 + (a*r*b*s*m*(exp(e) - 2 + a*b)) / (n*(exp(e)-1)**2)

    eq1 = diff(f_grr, a)
    eq2 = diff(f_grr, b)
    res = None
    try:
        res = nsolve([eq1, eq2], (a, b), (1, 1))
    except:
        res = nsolve([eq1, eq2], (a, b), (1, 1), verify=False)
    print(np.ceil(res[0]), np.ceil(res[1]))

    a, b = symbols('a b', real=True, positive=True)
    f = ((2*p*(a*r+b*s))/(a*b))**2 + (4*a*r*b*s*m*exp(e)) / (n*(exp(e)-1)**2)

    eq1 = diff(f, a)
    eq2 = diff(f, b)
    res = None
    try:
        res = nsolve([eq1, eq2], (a, b), (1, 1))
    except:
        res = nsolve([eq1, eq2], (a, b), (1, 1), verify=False)
    print(np.ceil(res[0]), np.ceil(res[1]))
    print("\n")
    #
    #
    # # cat / num
    b = 5
    ry = 1/5
    rx = 0.9
    a = symbols('a', real=True, positive=True)
    f_grr = ((2*p*b*ry)/(a*b))**2 + (a*rx*b*ry*m*(exp(e) - 2 + a*b)) / (n*(exp(e)-1)**2)

    eq1 = diff(f_grr, a)
    res = None
    try:
        res = nsolve(eq1, a,  (1, 100), solver='bisect')
    except:
        res = nsolve(eq1, a,  (1, 100), solver='bisect', verify=False)
    #print(np.ceil(res))

    a = symbols('a', real=True, positive=True)
    f_oue = ((2*p*b*ry)/(a*b))**2 + (4*a*rx*b*ry*m*exp(e)) / (n*(exp(e)-1)**2)

    eq1 = diff(f_oue, a)
    res = None
    try:
        res = nsolve(eq1, a, (1, 100), solver='bisect')
    except:
        res = nsolve(eq1, a, (1, 100), solver='bisect', verify=False)
    #print(np.ceil(res))
    #print("\n")

    # grr 1 D
    x = symbols('x', real=True, positive=True)
    f = (p / x) ** 2 + (x*r*m) * (x - 2 + exp(e)) / (n * (exp(e) - 1) ** 2)

    eq1 = diff(f, x)
    res = None
    try:
        res = nsolve(eq1, x, (1, 150), solver='bisect')
    except:
        res = nsolve(eq1, x, (1, 150), solver='bisect', verify=False)
    #print(np.ceil(res))

    x1 = n * (p ** 2) * ((exp(e) - 1) ** 2)
    x2 = 2 * m * r * exp(e)
    lx = (x1 / x2) ** (1/3)
    #print(np.ceil(lx))
    #print("\n")