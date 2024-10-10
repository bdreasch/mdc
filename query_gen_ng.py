import random
import sys
import numpy as np

import enum

class AttrTypeNG(enum.Enum):
    categorical = 1
    numerical = 2

class QueryAttrNode:
    def __init__(self, attr=-1, interval_length=-1, domain=None, attr_type=None,
                 args=None):
        self.args = args
        self.attr_domain = domain
        self.attr_index = attr
        self.attr_type = attr_type
        self.interval_length_ratio = None
        self.interval_length = None
        self.left_interval = None
        self.right_interval = None
        self.cat_value = None

        if attr_type == AttrTypeNG.numerical:
            self.interval_length_ratio = 1
            self.interval_length = interval_length
            if self.interval_length == -1:
                self.interval_length = self.attr_domain
            self.left_interval = 0
            self.right_interval = self.left_interval + self.interval_length - 1
        else:

            self.cat_value = np.random.randint(0, domain, 1)[0]  # uniform cat values

            # for loan
            # if self.attr_index == 2:
            #     self.cat_value = 0
            #
            # if self.attr_index == 6 or self.attr_index == 8:
            #     self.cat_value = 1

            self.left_interval = self.cat_value
            self.right_interval = self.cat_value
            #self.interval_length = 1

    def set_interval_length_ratio(self, interval_length_ratio=1.0):

        # for loan
        # if self.attr_index == 3:
        #     self.left_interval = 5
        #     self.right_interval = 52
        #     return
        # if self.attr_index ==

        self.interval_length_ratio = interval_length_ratio   # seletivity
        window_size = int(np.floor(self.attr_domain * self.interval_length_ratio))
        # change due to selectivity 0.1 results in query result 0
        self.left_interval = random.randint(0, self.attr_domain - window_size)
        #middle = (self.attr_domain // 2) - 1
        #self.left_interval = middle - (window_size // 2)
        self.right_interval = self.left_interval + window_size
        if self.right_interval >= self.attr_domain:
            self.right_interval = self.attr_domain - 1


class Query:
    def __init__(self, query_dimension=-1, attr_num=-1, domains_list=None,
                 attr_types_list=None,
                 args=None):
        self.args = args
        self.attr_types_list = attr_types_list
        self.domains_list = domains_list
        self.query_dimension = query_dimension
        self.attr_num = attr_num
        self.selected_attr_index_list = []
        #self.mixed_selection_list = []
        self.attr_index_list = [i for i in range(self.attr_num)]
        self.attr_node_list = []
        assert self.query_dimension <= self.attr_num
        self.real_answer = None
        self.domain_ratio = 0
        self.attr_index2letter = {i: f'a{i}' for i in range(self.args.attr_num)}
        self.initialize_query()
        self.set_selected_attr_list()



    def initialize_query(self):
        # for each attr in the dataset
        for i, d, t, in zip(range(self.attr_num), self.domains_list, self.attr_types_list):
            self.attr_node_list.append(QueryAttrNode(i, domain=d, attr_type=t, args=self.args))

    def set_selected_attr_list(self):


        # self.selected_attr_index_list = [random.randint(0, self.args.attr_num-1)]


        # self.selected_attr_index_list = [0]


        # grids = [[0, 3], [0, 5], [1, 3], [1, 5], [3, 4], [4, 5]]
        # self.selected_attr_index_list = grids[random.randint(0, len(grids) - 1)]


        # self.selected_attr_index_list = [0, 4, 5]


        arr = [0, 1, 2, 3, 4, 5]
        np.random.shuffle(arr)
        self.selected_attr_index_list = arr[:self.query_dimension]
        self.selected_attr_index_list.sort()


        # if self.args.query_dimension == 2:
        #     self.selected_attr_index_list = [0, 1]
        # elif self.args.query_dimension == 3:
        #     self.selected_attr_index_list = [0, 1, 3]
        # self.args.query_dimension == 4:
        #     self.selected_attr_index_list = [0, 1, 3, 4]
        # elif self.args.query_dimension == 5:
        #     self.selected_attr_index_list = [0, 1, 2, 3, 5]
        # elif self.args.query_dimension == 6:
        #     self.selected_attr_index_list = [0, 1, 2, 3, 4, 5]
        # elif self.args.query_dimension == 7:
        #     self.selected_attr_index_list = [0, 1, 2, 3, 4, 5, 7]
        # elif self.args.query_dimension == 8:
        #     self.selected_attr_index_list = [0, 1, 2, 3, 4, 5, 6, 7]
        # elif self.args.query_dimension == 9:
        #     self.selected_attr_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 9]
        # elif self.args.query_dimension == 10:
        #     self.selected_attr_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


    # define the range for each selected attr
    def define_values_for_selected_attrs(self, interval_length_ratios):

        for i in range(len(self.selected_attr_index_list)):
            index = self.selected_attr_index_list[i]
            node = self.attr_node_list[index]
            node.set_interval_length_ratio(interval_length_ratios[i])


    def print_query_answer(self, file_out=None):
        file_out.write(str(self.real_answer) + "\n")

    def print_query(self, file_out=None):

        len_attr = len(self.selected_attr_index_list)
        it = 0
        line = ""
        for i in self.selected_attr_index_list:
            qn = self.attr_node_list[i]
            if qn.attr_type == AttrTypeNG.numerical:
                line += str(qn.left_interval) + "<=" + self.attr_index2letter[qn.attr_index] + "<=" + str(qn.right_interval)
            elif qn.attr_type == AttrTypeNG.categorical:
                line += self.attr_index2letter[qn.attr_index] + "=" + str(qn.cat_value)
            else:
                raise Exception("print_query Invalid attr type")
            it += 1
            if it < len_attr:
                line += " and "
        file_out.write(line + "\n")
        #print('real_answer:', self.real_answer, end=" ", file=file_out)
        #print(file=file_out)


class QueryList:
    def __init__(self,
                 query_dimension=-1,
                 attr_num=-1,
                 query_num=-1,
                 dimension_query_volume=0.1,
                 attr_types_list=None,
                 args=None, domains_list=None):
        self.args = args
        self.attr_types_list = attr_types_list
        self.domains_list = domains_list
        self.query_dimension = query_dimension
        self.query_num = query_num
        self.attr_num = attr_num
        if self.attr_num == -1:
            self.attr_num = self.args.attr_num
        self.query_list = []
        self.real_answer_list = []
        self.dimension_query_volume = dimension_query_volume
        self.direct_multiply_MNAE = None
        self.max_entropy_MNAE = None
        self.weight_update_MNAE = None
        self.avg_select_vec = []
        self.avg_select = 0
        assert self.query_dimension <= self.attr_num and self.query_num > 0

    def generate_query_list(self):
        opts = [1, 2, 3, 4, 5, 6]
        dimensions = np.random.choice(opts, size=self.query_num, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
        for i in range(self.query_num):
            query = Query(dimensions[i],
                          self.attr_num,
                          domains_list=self.domains_list,
                          attr_types_list=self.attr_types_list,
                          args=self.args)
            selectivities = np.random.randint(low=7, high=9, size=dimensions[i])
            selectivities = selectivities * 0.1
            for s in selectivities:
                self.avg_select_vec.append(s)
            query.define_values_for_selected_attrs(selectivities)
            self.query_list.append(query)

        self.avg_select = sum(self.avg_select_vec) / len(self.avg_select_vec)

    def generate_real_answer_list(self, user_record):

        for query in self.query_list:
            count = 0
            for user_i in range(self.args.user_num):
                flag = True
                # for tmp_attr_node in tmp_range_query.attr_node_list:
                for attr_index in query.selected_attr_index_list:
                    attr_node = query.attr_node_list[attr_index]
                    try:
                        real_value = user_record[user_i][attr_index]
                    except Exception as ex:
                        print("generate_real_answer_list [ERROR]: user_i: {} attr_index: {}".format(user_i, attr_index))

                    if attr_node.attr_type == AttrTypeNG.numerical:
                        if attr_node.left_interval <= real_value <= attr_node.right_interval:
                            continue
                        else:
                            flag = False
                            break
                    elif attr_node.attr_type == AttrTypeNG.categorical:
                        if attr_node.cat_value == real_value:
                            continue
                        else:
                            flag = False
                            break
                if flag:
                    count += 1

            query.real_answer = count
            self.real_answer_list.append(count)
        return 0 in self.real_answer_list

    def print_query_list(self, file_out=None):
        for i in range(len(self.query_list)):
            tmp_query = self.query_list[i]
            file_out.write("select count(*) from foo where ")
            tmp_query.print_query(file_out)

    def print_query_answers(self, file_out=None):
        for i in range(len(self.query_list)):
            tmp_query = self.query_list[i]
            tmp_query.print_query_answer(file_out)

