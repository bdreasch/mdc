from scipy.stats import truncexpon, truncnorm
import matplotlib.pyplot as plt

def get_truncated_normal(count, upp, mean=0, sd=1, low=0):
    gen = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    return gen.rvs(count).astype(int)

acc = []
for i in range(100):
    attr_num = 24
    x = None
    while True:
        x = get_truncated_normal(3, attr_num, mean=attr_num//2, sd=int(attr_num/6))
        if len(x) == len(set(x)):
            break

    for j in x:
        acc.append(j)
plt.hist(acc)
plt.show()