import random
import sys
import numpy as np


class QueryAttrNode:
    def __init__(self, attr=-1, interval_length=-1, args=None):
        self.args = args
        self.attr_index = attr
        self.interval_length_ratio = 1
        self.interval_length = interval_length
        if self.interval_length == -1:  # use this to make the default value equal to self.args.domain_size
            self.interval_length = self.args.domain_size

        self.left_interval = 0
        self.right_interval = self.left_interval + self.interval_length - 1

        self.selected_left_interval_ratio = 0

    def set_interval_length_ratio(self, interval_length_ratio=1.0):
        self.interval_length_ratio = interval_length_ratio   # seletivity
        window_size = np.floor(self.args.domain_size * self.interval_length_ratio)
        self.left_interval = random.randint(0, self.args.domain_size - window_size) - 1
        self.right_interval = self.left_interval + window_size
        assert self.right_interval < self.args.domain_size


class RangeQuery:
    def __init__(self, query_dimension=-1, attr_num=-1, args=None):
        self.args = args
        self.query_dimension = query_dimension
        self.attr_num = attr_num
        self.selected_attr_index_list = []
        self.attr_index_list = [i for i in range(self.attr_num)]
        self.attr_node_list = []
        assert self.query_dimension <= self.attr_num
        self.real_answer = None
        self.domain_ratio = 0

    def initialize_range_query(self):
        for i in range(self.attr_num):
            self.attr_node_list.append(QueryAttrNode(i, args=self.args))

    def set_selected_attr_list(self):
        arr = self.attr_index_list.copy()
        np.random.shuffle(arr)
        self.selected_attr_index_list = arr[:self.query_dimension]
        self.selected_attr_index_list.sort()

    # define the range for each selected attr
    def set_query_attr_node_list_interval_length_ratio(self, interval_length_ratio=1.0):
        for i in self.selected_attr_index_list:
            self.attr_node_list[i].set_interval_length_ratio(interval_length_ratio)

    def print_range_query(self, file_out=None):
        if file_out is None:
            file_out = sys.stdout
        print("selected attrs:", self.selected_attr_index_list, end="\t\t: ", file=file_out)
        for qn in self.attr_node_list:
            print("[", qn.attr_index, "|",
                  qn.left_interval,
                  qn.right_interval, "]", end=" ", file=file_out)

        print('real_answer:', self.real_answer, end=" ", file=file_out)
        print(file=file_out)

    def Main(self):
        self.initialize_range_query()
        self.set_selected_attr_list()


class RangeQueryList:

    def __init__(self,
                 query_dimension=-1,
                 attr_num=-1,
                 query_num=-1,
                 dimension_query_volume=0.1,
                 args=None):
        self.args = args
        self.query_dimension = query_dimension
        self.query_num = query_num
        self.attr_num = attr_num
        if self.attr_num == -1:
            self.attr_num = self.args.attr_num
        self.range_query_list = []
        self.real_answer_list = []
        self.dimension_query_volume = dimension_query_volume
        self.direct_multiply_MNAE = None
        self.max_entropy_MNAE = None
        self.weight_update_MNAE = None
        assert self.query_dimension <= self.attr_num and self.query_num > 0

    def generate_range_query_list(self):
        # print("generating range queries...")
        for i in range(self.query_num):
            tmp_range_query = RangeQuery(self.query_dimension, self.attr_num, args=self.args)
            tmp_range_query.Main()
            tmp_range_query.set_query_attr_node_list_interval_length_ratio(self.dimension_query_volume)
            self.range_query_list.append(tmp_range_query)

    def generate_real_answer_list(self, user_record):
        # print("get real counts of queries...")
        for rq in self.range_query_list:
            count = 0
            for user_i in range(self.args.user_num):
                flag = True
                # for tmp_attr_node in tmp_range_query.attr_node_list:
                for attr_index in rq.selected_attr_index_list:
                    attr_node = rq.attr_node_list[attr_index]
                    real_value = user_record[user_i][attr_index]
                    if attr_node.left_interval <= real_value <= attr_node.right_interval:
                        pass
                    else:
                        flag = False
                        break
                if flag:
                    count += 1

            rq.real_answer = count
            self.real_answer_list.append(count)
        return

    def Main(self):
        self.generate_range_query_list()

    def print_range_query_list(self, file_out = None):
        for i in range(len(self.range_query_list)):
            tmp_range_query = self.range_query_list[i]
            tmp_range_query.print_range_query(file_out)

