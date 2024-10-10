import numpy as np
from scipy.stats import truncnorm

def get_truncated_normal(upp, mean=0, sd=1, low=0):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

n_l = []
# powers = np.arange(5.0, 7.1, 0.2, dtype=float)
# for i in powers:
#     n_l.append(int(10**i))

n_l = [1000000]

for n in n_l:
    num_user = n
    # dataset_config = [["n", 64],
    #                   ["n", 64]]

    dataset_config = [["n", 100],
                      ["n", 100],
                      ["n", 100],
                      ["n", 100],
                      ["n", 100],
                      ["n", 100]]

    with open("normal_n" + str(num_user) + "_num.txt", 'w') as writer:
        line = ""
        for z in range(len(dataset_config)):
            att = dataset_config[z]
            line += str(att[0])
            if z < (len(dataset_config) - 1):
                line += " "
            else:
                line += "\n"
                writer.write(line)

        line = ""
        for z in range(len(dataset_config)):
            att = dataset_config[z]
            line += str(att[1])
            if z < (len(dataset_config) - 1):
                line += " "
            else:
                line += "\n"
                writer.write(line)

        data = []
        for attr in dataset_config:
            gen = get_truncated_normal(mean=attr[1] / 2, sd=11, upp=attr[1])
            data.append(gen.rvs(num_user).astype(int))

        for column in data:
            print("Column size: {}".format(len(column)))
        for a in data:
            print("min: {} max: {}".format(min(a), max(a)))

        for i in range(num_user):
            values = []
            for j in range(len(dataset_config)):
                aux = data[j]
                values.append(aux[i])

            line = ""
            assert len(values) == len(dataset_config)
            for j in range(len(values)):
                v = values[j]
                line += str(v)
                if j < (len(values) - 1):
                    line += " "
            line += "\n"
            writer.write(line)