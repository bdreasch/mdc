import numpy as np
from scipy.stats import truncexpon

def get_truncated_exp(upper, scale, lower=0):
    return truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)

num_user = 1000000
dataset_config = [["c", 4],
                  ["n", 100],
                  ["c", 4],
                  ["n", 100],
                  ["c", 4],
                  ["n", 100]]

with open('exp_n1000000_mixed.txt', 'w') as writer:
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
        scale = 17
        if attr[0] == "c":
            scale = 1
        gen = get_truncated_exp(upper=attr[1], scale=scale)
        data.append(gen.rvs(num_user).astype(int))
    for a in data:
        print("min: {} max: {}".format(min(a), max(a)))

    for i in range(num_user):
        values = []
        for j in range(len(dataset_config)):
            aux = data[j]
            values.append(aux[i])

        line = ""
        for j in range(len(values)):
            v = values[j]
            line += str(v)
            if j < (len(values) - 1):
                line += " "
        line += "\n"
        writer.write(line)