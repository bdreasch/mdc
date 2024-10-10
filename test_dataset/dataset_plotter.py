import numpy as np
import math
from scipy.stats import truncexpon, truncnorm
import matplotlib.pyplot as plt

# def g(e, a, n, m):
#     x1 = 2 * a * (np.exp(e) - 1)
#     x2 = n / (m * np.exp(e))
#     x3 = math.sqrt(x2)
#     g2 = math.sqrt(x1 * x3)
#     return math.ceil(g2)
#
# a = 0.02
# ep = [1,2,3,4,5]
# ep = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
# for e in ep:
#     print(g(e, a, 1000000, 21))

# def get_truncated_normal(upp, mean=0, sd=1, low=0):
#     return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
#
#
# lower = 0
# upper = 100
# scale = 17
# mean = upper // 2
# #X = truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
# gen = get_truncated_normal(upp=upper, mean=mean, sd=11)
# data = gen.rvs(10000).astype(int)
# print("{} {}".format(max(data), min(data)))
#
# fig, ax = plt.subplots()
# ax.hist(data)
# plt.show()

# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
#
# def truncated_power_law(a, upper):
#     x = np.arange(1, upper+1, dtype='int')
#     pmf = 1/x**a
#     pmf /= pmf.sum()
#     return stats.rv_discrete(values=(range(0, upper), pmf))
#
# a = 1
# m = 100
# d = truncated_power_law(a=a, upper=m)
#
#
# sample = d.rvs(size=10000)
# print("{} {}".format(max(sample), min(sample)))
# plt.hist(sample, bins=np.arange(m)+0.5)
# plt.show()

def get_truncated_normal(upp, mean=0, sd=1, low=0):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

gen = get_truncated_normal(mean=2, sd=11, upp=6)
data = gen.rvs(1000000).astype(int)

print("min: {} max: {}".format(min(data), max(data)))