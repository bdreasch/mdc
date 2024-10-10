import math
from sympy import symbols, diff, exp, solve, nsolve
import numpy as np

kOUE = "OUE"
kGRR = "GRR"

class ChooseGranularityBeta:
    def __init__(self, attr_group_list, alpha1=0.7, alpha2=0.03, args=None):
        self.args = args
        self.alpha_1 = alpha1
        self.alpha_2 = alpha2
        self.attr_group_list = attr_group_list

    def get_grr_error(self, x, user_num=0):
        ep = self.args.epsilon
        m = len(self.attr_group_list)
        #n = user_num
        #return (np.exp(ep) + x - 2) / (n * (np.exp(ep) - 1) ** 2)

        n = self.args.user_num
        return (m*(np.exp(ep) + x - 2)) / (n * (np.exp(ep) - 1) ** 2)

    def get_oue_error(self, user_num=0):
        ep = self.args.epsilon
        m = len(self.attr_group_list)
        #n = user_num
        # return (4 * np.exp(ep)) / (n * (np.exp(ep) - 1) ** 2)

        n = self.args.user_num
        return (4*m*np.exp(ep)) / (n*(np.exp(ep) - 1)**2)

    def get_lx_oue(self, n_users=0):
        ep = self.args.epsilon
        r = self.args.rx
        m = len(self.attr_group_list)
        #n = n_users
        n = self.args.user_num
        x1 = n * (self.alpha_1**2) * ((math.exp(ep) - 1)**2)
        x2 = 2 * m * r * math.exp(ep)
        #x2 = 2 * r * math.exp(ep)
        lx = (x1/x2)**(1/3)
        return int(np.floor(lx))

    def get_lx_grr(self, n_users=0):
        ep = self.args.epsilon
        r = self.args.rx
        m = len(self.attr_group_list)
        n = self.args.user_num
        #n = n_users
        x = symbols('x', real=True, positive=True)
        #f = (self.alpha_1 / x) ** 2 + (x * r) * (x - 2 + exp(ep)) / (n * (exp(ep) - 1) ** 2)
        f = (self.alpha_1 / x) ** 2 + (x*r*m) * (x - 2 + exp(ep)) / (n * (exp(ep) - 1) ** 2)
        eq1 = diff(f, x)
        res = None
        try:
            res = nsolve(eq1, x, (1, 150), solver='bisect')
        except:
            res = nsolve(eq1, x, (1, 150), solver='bisect', verify=False)
        return int(np.floor(res))

    def get_lxly_nn_oue(self):
        e = self.args.epsilon
        rx = self.args.rx
        ry = self.args.ry
        m = len(self.attr_group_list)
        n = self.args.user_num
        a, b = symbols('a b', real=True, positive=True)
        f = ((2*self.alpha_2*(a*rx+b*ry))/(a*b))**2 + (4*a*rx*b*ry*m*exp(e)) / (n*(exp(e)-1)**2)
        eq1 = diff(f, a)
        eq2 = diff(f, b)
        res = None
        try:
            res = nsolve([eq1, eq2], (a, b), (1, 1))
        except:
            res = nsolve([eq1, eq2], (a, b), (1, 1), verify=False)
        lx = np.ceil(res[0])
        ly = np.ceil(res[1])
        return int(lx), int(ly)

    def get_lxly_nn_oue_error(self, lx, ly):
        e = self.args.epsilon
        rx = self.args.rx
        ry = self.args.ry
        m = len(self.attr_group_list)
        n = self.args.user_num
        error = ((2*self.alpha_2*(lx*rx + ly*ry))/(lx*ly))**2 + (4*lx*rx*ly*ry*m*np.exp(e)) / n*((np.exp(e) - 1)**2)
        return error

    def get_lxly_nn_grr(self):
        e = self.args.epsilon
        rx = self.args.rx
        ry = self.args.ry
        m = len(self.attr_group_list)
        n = self.args.user_num
        a, b = symbols('a b', real=True, positive=True)
        f = ((2*self.alpha_2*(a*rx+b*ry))/(a*b))**2 + (a*rx*b*ry*m*(exp(e) - 2 + a*b)) / (n*(exp(e)-1)**2)
        eq1 = diff(f, a)
        eq2 = diff(f, b)
        res = None
        try:
            res = nsolve([eq1, eq2], (a, b), (1, 1))
        except:
            res = nsolve([eq1, eq2], (a, b), (1, 1), verify=False)
        lx = np.ceil(res[0])
        ly = np.ceil(res[1])
        return int(lx), int(ly)

    def get_lxly_nn_grr_error(self, lx, ly):
        e = self.args.epsilon
        rx = self.args.rx
        ry = self.args.ry
        m = len(self.attr_group_list)
        n = self.args.user_num
        error = ((2*self.alpha_2*(lx*rx + ly*ry))/(lx*ly))**2 + (lx*rx*ly*ry*m*(np.exp(e) + lx*ly - 2)) / n*((np.exp(e) - 1)**2)
        return error

    def get_lxly_cn_oue(self, b, ry):
        e = self.args.epsilon
        rx = self.args.rx
        m = len(self.attr_group_list)
        n = self.args.user_num
        a = symbols('a', real=True, positive=True)
        f = ((2*self.alpha_2*ry)/(a))**2 + (4*a*rx*b*ry*m*exp(e)) / (n*(exp(e)-1)**2)
        eq1 = diff(f, a)
        res = None
        try:
            res = nsolve(eq1, a, (1, 100), solver='bisect')
        except:
            res = nsolve(eq1, a, (1, 100), solver='bisect', verify=False)
        lx = np.ceil(res)
        return int(lx)

    def get_lxly_cn_oue_error(self, lx, b, ry):
        e = self.args.epsilon
        rx = self.args.rx
        m = len(self.attr_group_list)
        n = self.args.user_num
        error = ((2*self.alpha_2*ry)/lx)**2 + (4*lx*rx*b*ry*m*np.exp(e)) / n*((np.exp(e) - 1)**2)
        return error

    def get_lxly_cn_grr(self, b, ry):
        e = self.args.epsilon
        rx = self.args.rx
        m = len(self.attr_group_list)
        n = self.args.user_num
        a = symbols('a', real=True, positive=True)
        f = ((2*self.alpha_2*ry)/a)**2 + (a*rx*b*ry*m*(exp(e) - 2 + a*b)) / (n*(exp(e)-1)**2)
        eq1 = diff(f, a)
        res = None
        try:
            res = nsolve(eq1, a, (1, 100), solver='bisect')
        except:
            res = nsolve(eq1, a, (1, 100), solver='bisect', verify=False)
        lx = np.ceil(res)
        return int(lx)

    def get_lxly_cn_grr_error(self, lx, b, ry):
        e = self.args.epsilon
        rx = self.args.rx
        m = len(self.attr_group_list)
        n = self.args.user_num
        error = ((2*self.alpha_2*ry)/lx)**2 + (lx*rx*b*ry*m(np.exp(e) + lx*b - 2)) / n*((np.exp(e) - 1)**2)
        return error

    def get_general_nn_oue(self, rx_vec, n_users):
        e = self.args.epsilon
        m = len(self.attr_group_list)
        #n = n_users
        n = self.args.user_num
        #rx_vec = [0.3, 0.5, 0.6]
        vas = [symbols('x%d' % i, real=True, positive=True) for i in range(len(rx_vec))]
        sum_rl = 0
        for i in range(len(rx_vec)):
            sum_rl += rx_vec[i] * vas[i]

        prod_rl = rx_vec[0] * vas[0]
        for i in range(1, len(rx_vec)):
            prod_rl *= rx_vec[i] * vas[i]

        prod_l = vas[0]
        for i in range(1, len(rx_vec)):
            prod_l *= vas[i]

        #f = ((len(rx_vec) * 0.3 * (sum_rl)) / (prod_l)) ** 2 + (4 * prod_rl * exp(e)) / (n * (exp(e) - 1) ** 2)
        f = ((len(rx_vec)*0.3*(sum_rl))/(prod_l))**2 + (4*prod_rl*m*exp(e)) / (n*(exp(e)-1)**2)

        equations = []
        for v in vas:
            equations.append(diff(f, v))

        tupe = tuple([1] * len(rx_vec))
        try:
            res = nsolve(equations, vas, tupe)
        except:
            res = nsolve(equations, vas, tupe, verify=False)

        sizes = []
        for r in res:
            sizes.append(np.floor(r))

        return sizes

    def get_general_nn_grr(self, rx_vec, n_users):
        e = self.args.epsilon
        m = len(self.attr_group_list)

        n = self.args.user_num
        #n = n_users
        vas = [symbols('x%d' % i, real=True, positive=True) for i in range(len(rx_vec))]
        sum_rl = 0
        for i in range(len(rx_vec)):
            sum_rl += rx_vec[i] * vas[i]

        prod_rl = rx_vec[0] * vas[0]
        for i in range(1, len(rx_vec)):
            prod_rl *= rx_vec[i] * vas[i]

        prod_l = vas[0]
        for i in range(1, len(rx_vec)):
            prod_l *= vas[i]

        #f = ((2 * 0.3 * sum_rl) / prod_l) ** 2 + (prod_rl * (exp(e) - 2 + prod_l)) / (n * (exp(e) - 1) ** 2)
        f = ((2 * 0.3 * sum_rl) / prod_l) ** 2 + (prod_rl * m * (exp(e) - 2 + prod_l)) / (n * (exp(e) - 1) ** 2)
        equations = []
        for v in vas:
            equations.append(diff(f, v))
        tupe = tuple([1] * len(rx_vec))
        try:
            res = nsolve(equations, vas, tupe)
        except:
            res = nsolve(equations, vas, tupe, verify=False)

        sizes = []
        for r in res:
            sizes.append(np.ceil(r))

        return sizes