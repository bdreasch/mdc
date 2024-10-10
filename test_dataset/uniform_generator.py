import numpy as np

n_l = [1000000]
# powers = np.arange(5.0, 7.1, 0.2, dtype=float)
# for i in powers:
#     n_l.append(int(10**i))


for n in n_l:
    num_user = n

    # dataset_config = [["n", 100],
    #                   ["n", 100],
    #                   ["n", 100],
    #                   ["n", 100],
    #                   ["n", 100],
    #                   ["n", 100]]

    dataset_config = [["n", 64],
                      ["n", 64]]

    with open("uniform_n" + str(num_user) + "_num.txt", 'w') as writer:
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

        for i in range(num_user):
            values = []
            for attr in dataset_config:
                values.append(np.random.randint(0, attr[1], 1)[0])

            line = ""
            for j in range(len(values)):
                v = values[j]
                line += str(v)
                if j < (len(values) - 1):
                    line += " "
            line += "\n"
            writer.write(line)