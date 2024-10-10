import itertools
import networkx as nx

def find_grids(correlations, num_attrs, dataset, eps, bn):
    path = "test_dataset/tmp/tmp_groups_finder_{}_{}_bn={}.txt".format(dataset, eps, bn)
    with open(path, "w") as file:
        for group in correlations:
            for i in range(len(group)):
                file.write(str(group[i]))
                if i < len(group) - 1:
                    file.write(",")
            file.write("\n")

    G = nx.read_edgelist(path, delimiter=',')
    #return nx.clique.find_cliques(G)


    grids = []
    for clq in nx.clique.find_cliques(G):
        grids.append(list(map(int, clq)))

    for i in range(num_attrs):
        grids.append([i])

    return grids

