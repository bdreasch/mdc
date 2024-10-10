
import math
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
import parameter_setting as para
import time
from sklearn import preprocessing
from sklearn.preprocessing import scale


def norm_sub(cell_est_list: list, user_num=None, tolerance=1):
    np_cell_est_list = np.array(cell_est_list)
    estimates = np.copy(np_cell_est_list)
    est_int = estimates.astype(int)
    while (np.fabs(sum(estimates) - user_num) > tolerance) or (estimates < 0).any() or est_int.sum() - user_num != 0:
        if (estimates <= 0).all():
            estimates[:] = user_num / estimates.size
            break
        estimates[estimates < 0] = 0
        total = sum(estimates)
        mask = estimates > 0
        diff = (user_num - total) / sum(mask)
        estimates[mask] += diff
        est_int = estimates.astype(int)
    return estimates

user_dist = np.random.randint(100, size=10)
print(user_dist)
print(scale(user_dist, axis=0, with_mean=True, with_std=True, copy=True) * 100)
# users = 100
# normalized_arr = preprocessing.normalize([user_dist])
# dist_array = normalized_arr * users
# dist_array = dist_array.astype(int)
# over = dist_array.sum() - users
# if over > 0:
#     ind = np.argpartition(dist_array, -over)[-over:]
#     for id in ind:
#         dist_array[id] -= 1
# elif over < 0:
#     ind = np.argpartition(dist_array, -over)[-over:]
#     for id in ind:
#         dist_array[id] += 1
#
#
# print(dist_array)
# print("sum: {}".format(dist_array.sum()))