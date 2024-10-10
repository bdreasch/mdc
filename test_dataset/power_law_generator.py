import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def truncated_power_law(a, upper):
    x = np.arange(1, upper+1, dtype='int')
    pmf = 1/x**a
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(0, upper), pmf))

num_user = 1000000
dataset_config = [["c", 6],
                  ["n", 100],
                  ["c", 6],
                  ["n", 100],
                  ["c", 6],
                  ["n", 100]]

with open('power_law_n1000000_mixed.txt', 'w') as writer:
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
        gen = truncated_power_law(a=1, upper=attr[1])
        data.append(gen.rvs(size=num_user))
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