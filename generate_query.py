import random
import sys
import numpy as np

import enum

from scipy.stats import truncexpon, truncnorm


class AttrType(enum.Enum):
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

        if attr_type == AttrType.numerical:
            self.interval_length_ratio = 1
            self.interval_length = interval_length
            if self.interval_length == -1:
                self.interval_length = self.attr_domain
            self.left_interval = 0
            self.right_interval = self.left_interval + self.interval_length - 1
        else:

            self.cat_value = np.random.randint(0, domain, 1)[0]  # uniform cat values



            self.left_interval = self.cat_value
            self.right_interval = self.cat_value
            #self.interval_length = 1

    def set_interval_length_ratio(self, interval_length_ratio=1.0):



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


    def draw_dims(self, qnt, exclude=[]):
        def get_truncated_normal(count, upp, mean=0, sd=1, low=0):
            gen = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
            return gen.rvs(count).astype(int)

        an = self.args.attr_num

        x = None
        while True:
            x = get_truncated_normal(qnt, an, mean=an//2, sd=an//6)
            if len(x) == len(set(x)):
                break

        return x.tolist()
    def workload0(self):


        self.selected_attr_index_list = self.draw_dims(self.args.query_dimension)

        #self.selected_attr_index_list = [0, 1, 2]
        #self.selected_attr_index_list = np.random.choice(self.args.attr_num, self.args.query_dimension, replace=False).tolist()
        # all random
        #dim_size = np.random.randint(low=1, high=self.args.query_dimension, size=1)[0]
        #self.selected_attr_index_list = np.random.choice(self.args.attr_num, dim_size, replace=False).tolist()


    def workload1(self):
        rnd = np.random.rand(1)[0]
        if rnd < 0.8:

            # any 1D
            self.selected_attr_index_list = self.draw_dims(1)
            if not self.selected_attr_index_list:
                raise Exception("selected_attr_index_list empty!")


        else:

            dim_size = np.random.randint(low=2, high=4, size=1)[0]
            self.selected_attr_index_list = self.draw_dims(dim_size)
            if not self.selected_attr_index_list:
                raise Exception("selected_attr_index_list empty!")

    def workload2(self):
        rnd = np.random.rand(1)[0]
        if rnd < 0.6:

            self.selected_attr_index_list = self.draw_dims(3)
        else:
            # any
            dim_size = np.random.randint(low=1, high=4, size=1)[0]
            self.selected_attr_index_list = self.draw_dims(dim_size)


    def workload3(self):
        rnd = np.random.rand(1)[0]
        if rnd < 0.9:
            self.selected_attr_index_list = np.random.choice(self.args.attr_num, 4, replace=False).tolist()
        else:
            # any
            dim_size = np.random.randint(3, size=1)[0] + 1
            # while dim_size == 4:
            #     dim_size = np.random.randint(self.query_dimension, size=1)[0] + 1

            self.selected_attr_index_list = self.draw_dims(dim_size)


    def workload4(self):
        rnd = np.random.rand(1)[0]
        if rnd < 0.7:
            self.selected_attr_index_list = [0, 1, 2, 3, 4, 5]
        else:
            # any
            dim_size = np.random.randint(low=1, high=4, size=1)[0]
            self.selected_attr_index_list = self.draw_dims(dim_size)

    def set_selected_attr_list(self):

        if self.args.dataset == "adult6_w1" or self.args.dataset == "loan6v4_w1" or self.args.dataset == "ipums6_w1" or self.args.dataset == "ipums12b_w1":
            self.workload1()
        elif self.args.dataset == "ipums6_w2" or self.args.dataset == "adult6_w2" or self.args.dataset == "adult12b_w2":
            self.workload2()
        elif self.args.dataset == "loan6v4_w3" or self.args.dataset == "adult6_w3" or self.args.dataset == "loan12b_w3":
            self.workload3()
        elif self.args.dataset == "bfive6_w4" or self.args.dataset == "bfive12b_w4":
            self.workload4()
        else:
            self.workload0()

        self.query_dimension = len(self.selected_attr_index_list)


    # define the range for each selected attr
    def define_values_for_selected_attrs(self, selecivity_list):
        #dummy = 0

        for i in self.selected_attr_index_list:
            node = self.attr_node_list[i]
            if node.attr_type == AttrType.numerical:

                node.set_interval_length_ratio(selecivity_list[i])


    def print_query_answer(self, file_out=None):
        file_out.write(str(self.real_answer) + "\n")

    def print_query(self, file_out=None):

        len_attr = len(self.selected_attr_index_list)
        it = 0
        line = ""
        for i in self.selected_attr_index_list:
            qn = self.attr_node_list[i]
            if qn.attr_type == AttrType.numerical:
                line += str(qn.left_interval) + "<=" + self.attr_index2letter[qn.attr_index] + "<=" + str(qn.right_interval)
            elif qn.attr_type == AttrType.categorical:
                line += self.attr_index2letter[qn.attr_index] + "=" + str(qn.cat_value)
            else:
                raise Exception("Invalid attr type")
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
                 dimension_query_volume_list=None,
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
        self.dimension_query_volume_list = dimension_query_volume_list
        self.direct_multiply_MNAE = None
        self.max_entropy_MNAE = None
        self.weight_update_MNAE = None
        assert self.query_dimension <= self.attr_num and self.query_num > 0

    def generate_query_list(self):
        for i in range(self.query_num):
            query = Query(self.query_dimension,
                          self.attr_num,
                          domains_list=self.domains_list,
                          attr_types_list=self.attr_types_list,
                          args=self.args)
            query.define_values_for_selected_attrs(self.dimension_query_volume_list)
            self.query_list.append(query)

    def generate_real_answer_list(self, user_record):

        teste=0
        for iq in range(len(self.query_list)):
            query = self.query_list[iq]
            count = 0
            for user_i in range(len(user_record)):
                flag = True
                # for tmp_attr_node in tmp_range_query.attr_node_list:
                for attr_index in query.selected_attr_index_list:
                    attr_node = query.attr_node_list[attr_index]
                    try:
                        real_value = user_record[user_i][attr_index]
                    except Exception as ex:
                        print("[ERROR]: user_i: {} attr_index: {}".format(user_i, attr_index))

                    if attr_node.attr_type == AttrType.numerical:
                        if attr_node.left_interval <= real_value <= attr_node.right_interval:
                            continue
                        else:
                            flag = False
                            break
                    # elif attr_node.attr_type == AttrType.categorical:
                    #     if attr_node.cat_value == real_value:
                    #         continue
                    #     else:
                    #         flag = False
                    #         break
                if flag:
                    count += 1

            query.real_answer = count
            if query.real_answer == 0:
                del self.query_list[iq]
                if len(self.query_list) == 0:
                    raise Exception("All queries are ZERO!")
                self.real_answer_list = []
                return True
            else:
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

