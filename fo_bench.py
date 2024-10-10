import frequency_oracle as fo
import numpy as np
import matplotlib.pyplot as plt

def plot_eps(eps, grr, oue, domain, n):
    xi = list(range(len(eps)))
    plt.plot(xi, grr, marker='o', label="GRR")
    plt.plot(xi, oue, marker='x', label="OUE")
    plt.xticks(xi, eps)
    plt.xlabel('Epsilon')
    plt.ylabel("MAE")
    plt.title("n={} d={}".format(n, domain))
    plt.yscale("log")
    plt.legend()
    plt.show()

def plot_domain(domain, grr, oue, eps, n):
    xi = list(range(len(domain)))
    plt.plot(xi, grr, marker='o', label="GRR")
    plt.plot(xi, oue, marker='x', label="OUE")
    plt.xticks(xi, domain)
    plt.xlabel('Domain')
    plt.ylabel("MAE")
    plt.title("n={} eps={}".format(n, eps))
    plt.yscale("log")
    plt.legend()
    plt.show()

def mae(real, predictions):
    real, predictions = np.array(real), np.array(predictions)
    return np.mean(np.abs(real - predictions))


n = 100000
domain = 30
dataset = np.random.randint(0, domain, n)
real = np.zeros(domain)
for item in dataset:
    real[item] += 1

eps = [1, 2, 3, 4, 5]

oue_error = []
grr_error = []

for e in eps:
    grr = fo.GRR(domain_size=domain, epsilon=e)
    oue = fo.OUE(domain_size=domain, epsilon=e)
    oue.group_user_num = n
    grr.group_user_num = n
    for item in dataset:
        oue.operation_perturb(item)
        grr.operation_perturb(item)
    oue.operation_aggregate()
    oue_est = oue.aggregated_count

    grr.operation_aggregate()
    grr_est = grr.aggregated_count

    oue_error.append(mae(real, oue_est))
    grr_error.append(mae(real, grr_est))

print(oue_error)
print(grr_error)
plot_eps(eps, grr_error, oue_error, domain, n)


# n = 100000
# domain = [16, 32, 64, 128, 256, 512]
# eps = 3
#
# oue_error = []
# grr_error = []
# repeat = 10
#
# for d in domain:
#     dataset = np.random.randint(0, d, n)
#     real = np.zeros(d)
#     for item in dataset:
#         real[item] += 1
#
#     grr = fo.GRR(domain_size=d, epsilon=eps)
#     oue = fo.OUE(domain_size=d, epsilon=eps)
#     oue.group_user_num = n
#     grr.group_user_num = n
#     for item in dataset:
#         oue.operation_perturb(item)
#         grr.operation_perturb(item)
#     oue.operation_aggregate()
#     oue_est = oue.aggregated_count
#
#     grr.operation_aggregate()
#     grr_est = grr.aggregated_count
#
#     oue_error.append(mae(real, oue_est))
#     grr_error.append(mae(real, grr_est))
#
# print(oue_error)
# print(grr_error)
# plot_domain(domain, grr_error, oue_error, eps, n)