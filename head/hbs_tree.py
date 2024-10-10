import treelib
import numpy as np
import math
from treelib import Tree, Node
import pandas as pd
import os 
from func_module import freqoracle
from func_module import errormetric
from func_module import realfreq
import ng.hilbert_curve as HC


# calculate theta
def theta_calculation(ahead_tree_height, epsilon, user_scale, branch_factor):
    user_scale_in_each_layer = user_scale / ahead_tree_height
    varience_of_OUE = 4 * math.exp(epsilon) / (user_scale_in_each_layer * (math.exp(epsilon) - 1) ** 2)
    return math.sqrt((branch_factor + 1) * varience_of_OUE)

# cria sub-nós
def construct_translation_vector(domain_size, branch_factor):
    translation_vector = []
    for i in range(branch_factor):
        left_interval = i * domain_size // branch_factor
        right_interval = i * domain_size // branch_factor
        translation_vector.append(np.array([left_interval, right_interval]))
    return translation_vector

# remove duplicated sub-domain partition vectors
def duplicate_remove(list1):
    list2 = []
    for li1 in list1:
        Flag1 = True

        for li2 in list2:
            if (li1 == li2).all():
                Flag1 = False
                break

        if Flag1 == True:
            list2.append(li1)

    return list2

# constroi um histograma para cada nível da árvore com o dataset real
def user_record_partition(data_path, ahead_tree_height, domain_size):
    dataset = np.loadtxt(data_path, np.int64)
    hilbert_indexes = np.zeros((len(dataset), ), dtype=np.int32)
    for i in range(len(dataset)):
        hilbert_indexes[i] = HC.to_hilbert_id(dataset[i])

    user_sample_id = np.random.randint(0, ahead_tree_height, len(dataset)).reshape(len(dataset), 1)  # user sample list
    user_histogram = np.zeros((ahead_tree_height, domain_size), dtype=np.int32)
    for k, item in enumerate(hilbert_indexes):
        user_histogram[user_sample_id[k], item] += 1
    return user_histogram

# cria os nós de um nível da árvore
def create_nodes(ahead_tree, tree_height, theta, branch,
                      translation_vector, layer_index):

    # id dos nós
    #         00
    #       10   11
    #     20 21 22 23

    for node in ahead_tree.leaves():
        if not node.data.divide_flag:
            continue

        elif (tree_height > 0 and node.data.divide_flag) and (node.data.frequency < theta):
            node.data.divide_flag = False
            continue

        else:
            # salva o intervalo ex: [0, 512] do nó corrente
            temp_item0 = np.zeros(node.data.interval.shape)
            for j in range(0, len(node.data.interval), 2):

                if node.data.interval[j + 1] - node.data.interval[j] > 1:
                    temp_item0[j] = node.data.interval[j]
                    temp_item0[j + 1] = (node.data.interval[j + 1] - node.data.interval[j]) // branch + node.data.interval[j]
                else:
                    temp_item0[j] = node.data.interval[j]
                    temp_item0[j + 1] = node.data.interval[j + 1]

            for item1 in translation_vector:
                node_name = str(tree_height) + str(layer_index)
                node_interval = temp_item0 + item1
                ahead_tree.create_node(node_name, node_name, parent=node.identifier, data=Nodex(0, True, 0, node_interval))
                layer_index += 1
                
# Step3: Noisy Frequency Construction (NFC) in Section 4.2
def ahead_tree_construction(ahead_tree, ahead_tree_height, theta, branch_factor,
                            translation_vector, user_dataset_partition, epsilon):
    tree_height = 0
    while tree_height < ahead_tree_height:
        layer_index = 0
        # update ahead_tree structrue 
        create_nodes(ahead_tree, tree_height, theta, branch_factor, translation_vector, layer_index)
        # update sub-domain partition vectors
        translation_vector[:] = translation_vector[:] // np.array([branch_factor, branch_factor])
        translation_vector = duplicate_remove(translation_vector)
        # update ahead_tree sub-domain frequency
        node_frequency_aggregation(ahead_tree, user_dataset_partition[tree_height], epsilon)
        tree_height += 1

# Step4: Post-processing (PP) in Section 4.2
def ahead_tree_postprocessing(ahead_tree):
    lowest_nodes_number = 0
    for _, node in reversed(list(enumerate(ahead_tree.all_nodes()))):
        if lowest_nodes_number < ahead_tree.size(ahead_tree.depth()):
            lowest_nodes_number += 1
            continue

        if ahead_tree.depth(node) != ahead_tree.depth() and ahead_tree.children(node.identifier) != []:
            numerator = 1 / node.data.count
            children_frequency_sum = 0
            for j, child_node in enumerate(ahead_tree.children(node.identifier)):
                numerator += 1 / child_node.data.count
                children_frequency_sum += child_node.data.frequency

            denominator = numerator + 1
            coeff0 = numerator / denominator
            coeff1 = 1 - coeff0

            node.data.frequency = coeff0 * node.data.frequency + coeff1 * children_frequency_sum
            node.data.count = 1/coeff0


def get_query_clusters(query_interval):
    query_left_interval_list = [int(query_interval[0]), int(query_interval[2])]
    query_right_interval_list = [int(query_interval[1]), int(query_interval[3])]

    q_hilbert_ids = []
    for i in range(query_left_interval_list[0], query_right_interval_list[0]):
        for j in range(query_left_interval_list[1], query_right_interval_list[1]):
            hi = HC.to_hilbert_id([i, j])
            q_hilbert_ids.append(hi)
    q_hilbert_ids.sort()

    groups = []
    c = [q_hilbert_ids[0]]
    for i in range(1, len(q_hilbert_ids)):
        if q_hilbert_ids[i] - q_hilbert_ids[i-1] == 1:
            c.append(q_hilbert_ids[i])
        else:
            groups.append(c)
            c = [q_hilbert_ids[i]]
            if i == len(q_hilbert_ids) - 1:
                groups.append(c)


    clusters = []
    for g in groups:
        init = g[0]
        end = g[-1]
        clusters.append([init, end])

    return clusters

# answer range queries
def ahead_tree_answer_query(ahead_tree, clusters, domain_size):
    estimated_frequency_value = 0
    # set 1-dim range query
    query_interval_temp = np.zeros(domain_size)
    for c in clusters:

        q_d1_left = c[0]
        q_d1_right = c[1]

        query_interval_temp[q_d1_left:q_d1_right+1] = 1
        query_interval_size = q_d1_right - q_d1_left + 1

        for i, node in enumerate(ahead_tree.all_nodes()):
            d1_left = int(node.data.interval[0])
            d1_right = int(node.data.interval[1])

            # not a leaf node
            if query_interval_temp.sum() and ahead_tree.children(node.identifier) != [] and \
                    query_interval_temp[d1_left:d1_right].sum() == (d1_right-d1_left):
                estimated_frequency_value += node.data.frequency
                query_interval_temp[d1_left:d1_right] = 0
                continue

            # leaf node
            # assume uniform distribution
            if query_interval_temp.sum() and ahead_tree.children(node.identifier) == []:
                intersection_ratio = query_interval_temp[d1_left:d1_right].sum()/(d1_right-d1_left)
                estimated_frequency_value += intersection_ratio * node.data.frequency
                query_interval_temp[d1_left:d1_right] = 0

    return estimated_frequency_value

# record query errors
def ahead_tree_query_error_recorder(ahead_tree, real_frequency, query_interval_table, domain_size, MAEDict):
    errList = np.zeros(len(query_interval_table))
    for query_id, query_interval in enumerate(query_interval_table):
        real_frequency_value = real_frequency[query_id]
        clusters = get_query_clusters(query_interval)
        estimated_frequency_value = ahead_tree_answer_query(ahead_tree, clusters, domain_size)
        errList[query_id] = real_frequency_value - estimated_frequency_value
        # d1_left = int(query_interval[0])
        # d1_right = int(query_interval[1])
        # print('answer index {}-th query'.format(query_id))
        # print("real_frequency_value: ", real_frequency_value)
        # print("estimated_frequency_value: ", estimated_frequency_value)

    MAEDict['rand'].append(errormetric.MSE_metric(errList))


def node_frequency_aggregation(ahead_tree, user_dataset, epsilon):
    # estimate the frequency values, and update the frequency values of the nodes
    p = 0.5
    q = 1.0 / (1 + math.exp(epsilon))

    user_record_list = []  # lista de count real de cada nó no nível corrente
    for node in ahead_tree.leaves():
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        user_record_list.append(user_dataset[d1_left:d1_right].sum())

    noise_vector = freqoracle.OUE_Noise(epsilon, np.array(user_record_list, np.int32), sum(user_record_list))
    noisy_frequency = freqoracle.Norm_Sub(noise_vector, len(noise_vector), sum(user_record_list), p, q)

    for i, node in enumerate(ahead_tree.leaves()):
        if node.data.count == 0:
            node.data.frequency = noisy_frequency[i]
            node.data.count += 1
        else:
            node.data.frequency = ((node.data.count * node.data.frequency) + noisy_frequency[i]) / (node.data.count + 1)
            node.data.count += 1


class Nodex(object):
    def __init__(self, frequency, divide_flag, count, interval):
        self.frequency = frequency
        self.divide_flag = divide_flag
        self.count = count
        self.interval = interval

def main_func(repeat_time, domain_size, branch_factor, ahead_tree_height, theta, real_frequency,
              query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name):
    MAEDict = {'rand': []}
    repeat = 0
    while repeat < repeat_time:
        # user partition
        user_dataset_partition = user_record_partition(data_path, ahead_tree_height, domain_size)

        # initialize the tree structure, set the root node
        ahead_tree = Tree()
        ahead_tree.create_node('Root', 'root', data=Nodex(1, True, 1, np.array([0, domain_size])))

        # construct sub-domain partition vectors
        translation_vector = construct_translation_vector(domain_size, branch_factor)

        # build a tree structure
        ahead_tree_construction(ahead_tree, ahead_tree_height, theta, branch_factor,
                                translation_vector, user_dataset_partition, epsilon)

        # ahead_tree post-processing
        ahead_tree_postprocessing(ahead_tree)

        # ahead_tree answer query
        ahead_tree_query_error_recorder(ahead_tree, real_frequency, query_interval_table, domain_size, MAEDict)

        # record errors
        #print(MSEDict)
        # MSEDict_temp = pd.DataFrame.from_dict(MSEDict, orient='columns')
        # MSEDict_temp.to_csv('rand_result/MSE_lle_ahead_branch{}-{}-{}-{}-{}-{}.csv'.format(branch,
        #                                                                                    data_name,
        #                                                                                    data_size_name,
        #                                                                                    domain_name,
        #                                                                                    epsilon,
        #                                                                                    repeat_time))
        repeat += 1
        #print("repeat time: ", repeat)

    #print("AVG MSE: {}".format(sum(MSEDict['rand']) / len(MSEDict['rand'])))
    #print("Dump mse: \n{}".format(MSEDict['rand']))
    return sum(MAEDict['rand']) / len(MAEDict['rand'])


def run_eps(epsilon):

    # set the number of repeated experiments
    repeat_time = 5

    # set data_dimension, branch and domain_size
    data_dimension = 2
    branch_factor = 3
    domain_size = 2 ** 6
    hilbert_domain = 2 ** 12
    ahead_tree_height = int(math.log(hilbert_domain, branch_factor))

    # load query table
    #query_path = './query_table/2d_normal_n1000000.txt'
    query_path = './query_table/2d_normal_skinny.txt'
    query_interval_table = np.loadtxt(query_path, int)
    #print("the top 5 range queries in query_interval_table: \n", query_interval_table[:1])

    # select dataset
    #data_name = '2d_num_normal'
    data_name = '2d_num_normal_skinny'
    data_size_name = 'set_10_5' \
                     ''
    domain_name = 'domain6_attribute{}'.format(data_dimension)

    # load dataset
    # data_path = './dataset/{}-{}-{}-data.txt'.format(data_name, data_size_name, domain_name)
    data_path = './dataset/2d_normal_n1000000_num.txt'
    dataset = np.loadtxt(data_path, np.int32)
    #print("the shape of dataset: ", dataset.shape)
    data_size = dataset.shape[0]

    # calculate/load true frequency
    real_frequency_path = './query_table/real_frequency-{}-{}-{}.npy'.format(data_name, data_size_name, domain_name)
    if os.path.exists(real_frequency_path):
        real_frequency = np.load(real_frequency_path)
    else:
        # fix the [query_interval_table] if there is more than one query
        real_frequency = realfreq.real_frequency_generation(dataset, data_size, list(np.ones(data_dimension, dtype=int) * domain_size),
                                                            data_dimension, query_interval_table)
        np.save(real_frequency_path, real_frequency)

    # calculate theta
    theta = theta_calculation(ahead_tree_height, epsilon, data_size, branch_factor)

    # running
    # fix the [query_interval_table] if there is more than one query
    mae = main_func(repeat_time, hilbert_domain, branch_factor, ahead_tree_height, theta, real_frequency, query_interval_table,
              epsilon, data_path, data_name, data_size_name, domain_name)
    print("{:.12f}".format(mae))


if __name__ == "__main__":
    eps = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for e in eps:
        run_eps(e)