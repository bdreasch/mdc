import math
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
import parameter_setting as para
import time
from sklearn import preprocessing

def setup_args(args=None):
    args.user_num = 2260701
    args.n_code = "1m"
    args.attr_num = 48

    # args.epsilon = 0.2
    args.dimension_query_volume = 0.9
    args.rx = args.dimension_query_volume
    args.ry = args.dimension_query_volume
    args.query_num = 100
    args.query_dimension = 3
    args.selectivity = True
    args.split_ratio = 0.01
    args.bn_degree = 0
    args.smart_post = False
    args.kd = True
    args.consist = True

kOUE = "OUE"
kGRR = "GRR"

class UserDistOptmizer:
    def __init__(self, rs_list, ls_list, grids_weights, args):
        self.rs_list = rs_list  # lista de listas
        self.ls_list = ls_list  # lista de listas
        self.grids_weights = grids_weights
        self.args = args

    def var_grr_1d(self, n, r, l_vec):
        ee = math.exp(self.args.epsilon)
        l = l_vec[0]
        nue = (0.7 / l) ** 2
        #print("nue 1d: {}".format(nue))
        ne = ((l * r * (ee + l - 2)) / (n * ((ee - 1) ** 2)))
        #print(ne)
        #print("ratio 1d {}".format(nue / ne))
        return nue + ne

    def var_oue_1d(self, n, r, l_vec):
        ee = math.exp(self.args.epsilon)
        l = l_vec[0]
        return (0.7 / l) ** 2 + ((4 * l * r * ee) / (n * (ee - 1) ** 2))

    def var_general_1d(self, n, r, l_vec):
        ee = math.exp(self.args.epsilon)
        l = l_vec[0]
        if 3*ee + 2 > l:
            return self.var_grr_1d(n, r, l_vec)
        return self.var_oue_1d(n, r, l_vec)

    def var_grr_nd(self, n, rx_vec, l_vec):
        ee = math.exp(self.args.epsilon)
        sum_rl = 0
        for i in range(len(rx_vec)):
            sum_rl += rx_vec[i] * l_vec[i]

        prod_rl = rx_vec[0] * l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_rl *= rx_vec[i] * l_vec[i]

        prod_l = l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_l *= l_vec[i]

        d = len(l_vec)
        nue = (((2**(d-1) * 0.03 * sum_rl) / prod_l) ** 2)
        #print("nue {}d: {}".format(len(l_vec), nue))
        ne = ((prod_rl * (ee + prod_l - 2)) / (n * ((ee - 1) ** 2)))
        #print(ne)
        #print("ratio nd {}".format(nue / ne))

        return nue + ne

    def var_oue_nd(self, n, rx_vec, l_vec):
        ee = math.exp(self.args.epsilon)
        sum_rl = 0
        for i in range(len(rx_vec)):
            sum_rl += rx_vec[i] * l_vec[i]

        prod_rl = rx_vec[0] * l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_rl *= rx_vec[i] * l_vec[i]

        prod_l = l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_l *= l_vec[i]

        return ((len(rx_vec) * 0.03 * (sum_rl)) / (prod_l)) ** 2 + (4 * prod_rl * ee) / (n * (ee - 1) ** 2)


    def var_general_nd(self, n, rx_vec, l_vec):
        ee = math.exp(self.args.epsilon)
        prod_l = l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_l *= l_vec[i]
        if 3*ee + 2 > prod_l:
            return self.var_grr_nd(n, rx_vec, l_vec)
        return self.var_oue_nd(n, rx_vec, l_vec)

    def find_user_dist(self):
        def fobj(ns):
            cost = 0
            for i in range(len(ns)):
                var_error = 0
                if len(self.ls_list[i]) == 1:
                    var_error = self.var_general_1d(ns[i], self.args.dimension_query_volume, self.ls_list[i])
                else:
                    rx_vec = [self.args.dimension_query_volume] * len(self.ls_list[i])
                    var_error = self.var_general_nd(ns[i], rx_vec, self.ls_list[i])

                cost += (self.grids_weights[len(self.ls_list[i])-1]**2) * var_error

            return cost


        # initial solution
        uniform_d = int(self.args.user_num / len(self.ls_list))
        ns0 = [uniform_d] * len(self.ls_list)

        # define bounds
        b = (0, (self.args.user_num//7))
        bnds = [b] * len(self.ls_list)

        def constr_fun(x):
            return np.array(sum(x))

        nlc = NonlinearConstraint(constr_fun, 0, self.args.user_num+1)
        sol = differential_evolution(fobj,
                                     bounds=bnds,
                                     #popsize=50,
                                     strategy="rand1bin",
                                     mutation=(0.3, 1.0),
                                     recombination=0.7,
                                     tol=1e-8,
                                     #maxiter=20,
                                     updating='deferred',
                                     #workers=-1,
                                     polish=False,
                                     #vectorized=True,
        constraints=nlc)


        user_dist = np.zeros(len(sol.x))
        for i in range(len(sol.x)):
            user_dist[i] = sol.x[i]
        # remaining = self.args.user_num - sum(user_dist)
        # user_dist[0] += remaining

        for i in range(len(user_dist)):
            w = self.grids_weights[i]
            if w == 0:
                user_dist[i] = 0

        sum_dist = user_dist.sum()

        cc = 0
        for i in range(user_dist.size):
            if user_dist[i] > 0:
                cc += 1
        over = sum_dist - self.args.user_num
        if over > 0:
            ratio = sum_dist / self.args.user_num
            if ratio > 1:
                user_dist /= ratio

            user_dist = user_dist.astype(int)

            over_residual = user_dist.sum() - self.args.user_num

            if over_residual > 0:
                high = 0
                high_i = 0
                for j in range(len(user_dist)):
                    if user_dist[j] > high:
                        high = user_dist[j]
                        high_i = j
                user_dist[high_i] -= over_residual
            elif over_residual < 0:
                for i in range(user_dist.size-1, user_dist.size+over_residual-1, -1):
                    user_dist[i] += 1

            over_residual = user_dist.sum() - self.args.user_num
        elif over < 0:
            #print("s3")
            over = over * -1
            non_zero = 0
            for udi in user_dist:
                if udi > 0:
                    non_zero += 1
            add_per_group = over // non_zero
            for i in range(len(user_dist)):
                if user_dist[i] > 0:
                    user_dist[i] += add_per_group

            user_dist = user_dist.astype(int)
            sum_dist = sum(user_dist)
            under_residual = self.args.user_num - sum_dist

            if under_residual > 0:
                user_dist[0] += under_residual
        else:
            user_dist = user_dist.astype(int)

        return user_dist.tolist()
