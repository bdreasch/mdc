import numpy as np
import math
import grid_generator as GridGen
from estimate_method import EstimateMethod
import frequency_oracle as FO
from frequency_oracle import FOProtocol
import itertools
import choose_granularity

import bin_packing_v2 as bin_packing
from generate_query import AttrType

kOUE = "OUE"
kGRR = "GRR"

class AGUniformGrid12wayOptimal:
    def __init__(self, domain_list=None, attr_type_list=None, args=None,
                 protocol=None, alpha1=0.7, alpha2=0.03, default=True):
        self.args = args
        self.group_attr_num = 2  # to construct 2-D grids
        self.group_num = 0
        self.domain_list = domain_list  # all attr domains
        self.attr_type_list = attr_type_list  # all attr domains
        self.attr_group_list = []  # attr_group
        self.grid_set = []
        self.answer_list = []
        self.weighted_update_answer_list = []
        self.default = default

        self.user_group_ldp_mech_list = []  # LDP mechanism for each attr group
        self.group_id2grid_size = {}
        self.group2d_id2mapping_lens = {}  # {attr_id, [g1_cell_len_list, g2_cell_len_list, g2index2g1_lens_int_index, g2index2g1_lens_sum, g2index2g1_lens_list]}
        self.group1d_id2cell_lens = {}
        self.protocol = protocol
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.group_id2proto = {}
        self.g1_list = []
        self.g2_list = []
        self.attr2cell_lens = {}
        self.best_protocol = kGRR

    def generate_attr_group(self):
        attr_group_list = []
        attr_index_list = [i for i in range(self.args.attr_num)]
        for attr_index in attr_index_list:
            attr_group_list.append((attr_index,))
        attr_group_2_way_list = list(itertools.combinations(attr_index_list,
                                                            self.group_attr_num))
        for g in attr_group_2_way_list:
            attr_group_list.append(g)


        if not self.default:
        # remove the 1D categorical groups
            for i in range(len(self.attr_type_list)):
                a_t = self.attr_type_list[i]
                if a_t == AttrType.categorical:
                    attr_group_list.remove((i,))

        self.group_num = len(attr_group_list)
        self.args.group_num = self.group_num
        self.attr_group_list = attr_group_list
        for i in range(len(self.attr_group_list)):
            self.attr_group_list[i] = list(self.attr_group_list[i])

        #print("FELIP g: {} u_per_g: {}".format(self.group_num, int(self.args.user_num / self.group_num)))
        return

    def define_initial_cell_list_lens(self, domain, gran):
        remaining_domain = domain
        remaining_g = gran
        cell_length_list = []
        while remaining_g > 0:
            cur_cell_len = remaining_domain // remaining_g
            cell_length_list.append(cur_cell_len)
            remaining_domain -= cur_cell_len
            remaining_g -= 1
        return cell_length_list

    def construct_g1_and_g2_list(self, attr_type, domain, attr_id):
        if attr_type == AttrType.numerical:
            g1_cell_len_list = self.define_initial_cell_list_lens(domain, self.g1_list[attr_id])
            g2_cell_len_list = self.define_initial_cell_list_lens(domain, self.g2_list[attr_id])
        elif attr_type == AttrType.categorical:
            g1_cell_len_list = [1] * domain
            g2_cell_len_list = [1] * domain
        else:
            raise Exception("Invalid attr type")
        return g1_cell_len_list, g2_cell_len_list

    def setup_all_grid_formatting(self):

        # make the matching
        for attr_id in range(len(self.domain_list)):
            d = self.domain_list[attr_id]
            t = self.attr_type_list[attr_id]

            find = True
            while find:

                l1, l2 = self.construct_g1_and_g2_list(t, d, attr_id)

                # l3 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                #       2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
                # l4 = [6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]
                #
                # sim1 = l1 == l3
                # sim2 = l2 == l4

                l3 = []
                for item in l1:
                    l3.append(int(item))

                l4 = []
                for item in l2:
                    l4.append(int(item))

                g2index2g1_lens_int_index = None
                g2index2g1_lens_sum = None
                g2index2g1_lens_list = None
                g1_len_list = None

                try:
                    if t == AttrType.numerical:
                        g2index2g1_lens_int_index, \
                        g2index2g1_lens_sum, \
                        g2index2g1_lens_list, \
                        g1_len_list = bin_packing.solve_bin_packing(l3, l4)
                    else:
                        # categorical
                        g1_len_list = l3

                except Exception as ex:
                    print("Error bin_packing attr_id: {}".format(attr_id))
                # if len(g1_len_list) != len(g1_cell_len_list):
                #     print("WARNING: Error bin_packing attr_id={}".format(attr_id))
                if len(g1_len_list) == len(l3):
                    find = False
                else:
                    # print("bin_packing not optimal for attr_id={}".format(attr_id))
                    self.g1_list[attr_id] += 1

            self.attr2cell_lens[attr_id] = [g1_len_list, l4,
                                            g2index2g1_lens_int_index,
                                            g2index2g1_lens_sum,
                                            g2index2g1_lens_list]
        return

    def calculate_inital_grid_2d_size(self, index):
        gsize_calculator = choose_granularity.ChooseGranularityBeta(attr_group_list=self.attr_group_list,
                                                                         args=self.args, alpha1=self.alpha1, alpha2=self.alpha2)
        group = self.attr_group_list[index]

        attr1_id = group[0]
        attr2_id = group[1]

        attr1_type = self.attr_type_list[attr1_id]
        attr2_type = self.attr_type_list[attr2_id]

        d1 = self.domain_list[attr1_id]
        d2 = self.domain_list[attr2_id]

        if attr1_type == AttrType.categorical and attr2_type == AttrType.categorical:
            self.group_id2grid_size[index] = [self.domain_list[attr1_id],
                                              self.domain_list[attr2_id]]
        elif attr1_type == AttrType.numerical and attr2_type == AttrType.numerical:
            grr_lx, grr_ly = gsize_calculator.get_lxly_nn_grr()
            if grr_lx > d1:
                grr_lx = d1
            if grr_ly > d2:
                grr_ly = d2
            oue_lx, oue_ly = gsize_calculator.get_lxly_nn_oue()
            if oue_lx > d1:
                oue_lx = d1
            if oue_ly > d2:
                oue_ly = d2

            grr_error = gsize_calculator.get_grr_error(grr_lx * grr_ly)
            oue_error = gsize_calculator.get_oue_error()

            if self.protocol == FOProtocol.ADAPTIVE:
                aux = 3 * np.exp(self.args.epsilon) + 2
                #if grr_lx * grr_ly < aux and grr_error < oue_error:
                if grr_error < oue_error:
                    self.group_id2grid_size[index] = [grr_lx, grr_ly]
                else:
                    self.group_id2grid_size[index] = [oue_lx, oue_ly]
            elif self.protocol == FOProtocol.GRR:
                self.group_id2grid_size[index] = [grr_lx, grr_ly]
            elif self.protocol == FOProtocol.OUE:
                self.group_id2grid_size[index] = [oue_lx, oue_ly]

        elif attr1_type == AttrType.categorical and attr2_type == AttrType.numerical:
            attr1_domain = self.domain_list[attr1_id]
            attr2_domain = self.domain_list[attr2_id]
            grr_ly = gsize_calculator.get_lxly_cn_grr(b=attr1_domain, ry=1/attr1_domain)
            if grr_ly > attr2_domain:
                grr_ly = attr2_domain
            oue_ly = gsize_calculator.get_lxly_cn_oue(b=attr1_domain, ry=1/attr1_domain)
            if oue_ly > attr2_domain:
                oue_ly = attr2_domain

            grr_error = gsize_calculator.get_grr_error(attr1_domain * grr_ly)
            oue_error = gsize_calculator.get_oue_error()

            if self.protocol == FOProtocol.ADAPTIVE:
                aux = 3 * np.exp(self.args.epsilon) + 2
                #if attr1_domain * grr_ly < aux and grr_error < oue_error:
                if grr_error < oue_error:
                    self.group_id2grid_size[index] = [attr1_domain, grr_ly]
                else:
                    self.group_id2grid_size[index] = [attr1_domain, oue_ly]
            elif self.protocol == FOProtocol.GRR:
                self.group_id2grid_size[index] = [attr1_domain, grr_ly]
            elif self.protocol == FOProtocol.OUE:
                self.group_id2grid_size[index] = [attr1_domain, oue_ly]

        elif attr1_type == AttrType.numerical and attr2_type == AttrType.categorical:
            attr2_domain = self.domain_list[attr2_id]
            attr1_domain = self.domain_list[attr1_id]
            grr_lx = gsize_calculator.get_lxly_cn_grr(b=attr2_domain, ry=1/attr2_domain)
            if grr_lx > attr1_domain:
                grr_lx = attr1_domain
            oue_lx = gsize_calculator.get_lxly_cn_oue(b=attr2_domain, ry=1/attr2_domain)
            if oue_lx > attr1_domain:
                oue_lx = attr1_domain


            grr_error = gsize_calculator.get_grr_error(attr2_domain * grr_lx)
            oue_error = gsize_calculator.get_oue_error()

            if self.protocol == FOProtocol.ADAPTIVE:
                aux = 3 * np.exp(self.args.epsilon) + 2
                #if grr_lx * attr2_domain < aux and grr_error < oue_error:
                if grr_error < oue_error:
                    self.group_id2grid_size[index] = [grr_lx, attr2_domain]
                else:
                    self.group_id2grid_size[index] = [oue_lx, attr2_domain]
            elif self.protocol == FOProtocol.GRR:
                self.group_id2grid_size[index] = [grr_lx, attr2_domain]
            elif self.protocol == FOProtocol.OUE:
                self.group_id2grid_size[index] = [oue_lx, attr2_domain]


    def calculate_inital_grid_1d_size(self, attr_id):
        gsize_calculator = choose_granularity.ChooseGranularityBeta(attr_group_list=self.attr_group_list,
                                                                         args=self.args, alpha1=self.alpha1, alpha2=self.alpha2)
        size_1d = 0
        attr_type = self.attr_type_list[attr_id]
        if attr_type == AttrType.categorical:
            size_1d = [self.domain_list[attr_id]]
        else:  # numerical
            d = self.domain_list[attr_id]
            lx_grr = gsize_calculator.get_lx_grr()
            if lx_grr > d:
                lx_grr = d
            lx_oue = gsize_calculator.get_lx_oue()
            if lx_oue > d:
                lx_oue = d

            grr_error = gsize_calculator.get_grr_error(lx_grr)
            oue_error = gsize_calculator.get_oue_error()

            if self.protocol == FOProtocol.ADAPTIVE:
                aux = 3 * np.exp(self.args.epsilon) + 2
                #if lx_grr < aux and grr_error < oue_error:
                if grr_error < oue_error:
                    size_1d = [lx_grr]
                else:
                    size_1d = [lx_oue]
            elif self.protocol == FOProtocol.GRR:
                size_1d = [lx_grr]
            elif self.protocol == FOProtocol.OUE:
                size_1d = [lx_oue]

        if self.default:
            self.group_id2grid_size[attr_id] = size_1d
        self.g1_list.append(size_1d[0])

    def optimize_grid_sizes(self):

        for a in range(self.args.attr_num):
            if self.attr_type_list[a] == AttrType.categorical:
                self.g2_list.append(self.domain_list[a])
                continue

            relevant_attr_group_list = []
            for i in range(self.group_num):
                group = self.attr_group_list[i]

                if a in group:
                    if len(group) == 2:
                        relevant_attr_group_list.append(i)

            a_sizes = []
            for rag in relevant_attr_group_list:
                group = self.attr_group_list[rag]
                sub_attr_index_in_grid = group.index(a)
                other_attr_index = 1 - sub_attr_index_in_grid
                grid_size = self.group_id2grid_size[rag]
                a_sizes.append(grid_size[sub_attr_index_in_grid])

            unified_size = (sum(a_sizes) // len(a_sizes)) - 1
            if unified_size <= 1:
                unified_size = 2
            self.g2_list.append(unified_size)
            for rag in relevant_attr_group_list:
                group = self.attr_group_list[rag]
                sub_attr_index_in_grid = group.index(a)
                self.group_id2grid_size[rag][sub_attr_index_in_grid] = unified_size

        return

    def construct_grid_set(self):

        for i in range(len(self.domain_list)):
            self.calculate_inital_grid_1d_size(i)

        for i in range(self.group_num):
            group = self.attr_group_list[i]
            if len(group) == 2:
                self.calculate_inital_grid_2d_size(i)

        self.optimize_grid_sizes()

        self.setup_all_grid_formatting()

        tmp_grid = None
        for i in range(self.group_num):
            group = self.attr_group_list[i]
            group_domain = []
            for attr in group:
                group_domain.append(self.domain_list[attr])
            if len(self.attr_group_list[i]) == 1:  # 1D grid

                d1_len_list = self.attr2cell_lens[attr][0]  # grab g1_len_list
                grid_attr_types = [self.attr_type_list[group[0]]]
                gran_list = [self.g1_list[group[0]]]
                tmp_grid = GridGen.UniformGrid(index=i,
                                               attr_set=group,
                                               domain_list=group_domain,
                                               attr_types_list=grid_attr_types,
                                               grid_size=gran_list,
                                               cell_length_d1_list=d1_len_list,
                                               args=self.args)
            else:
                type_dim_1 = self.attr_type_list[group[0]]
                type_dim_2 = self.attr_type_list[group[1]]
                grid_attr_types = [type_dim_1, type_dim_2]
                # for cat/cat and num/num grab g2 lens
                d1_len_list = self.attr2cell_lens[group[0]][1]  # grab g2_len_list
                d2_len_list = self.attr2cell_lens[group[1]][1]  # grab g2_len_list

                # if cat/cat or num/num
                gran_list = [self.g2_list[group[0]], self.g2_list[group[1]]]

                if type_dim_1 == AttrType.categorical and type_dim_2 == AttrType.numerical:
                    gran_list = [self.g2_list[group[0]], self.g2_list[group[1]]]
                elif type_dim_1 == AttrType.numerical and type_dim_2 == AttrType.categorical:
                    gran_list = [self.g2_list[group[0]], self.g2_list[group[1]]]

                tmp_grid = GridGen.UniformGrid(index=i,
                                               attr_set=group,
                                               domain_list=group_domain,
                                               attr_types_list=grid_attr_types,
                                               grid_size=gran_list,
                                               cell_length_d1_list=d1_len_list,
                                               cell_length_d2_list=d2_len_list,
                                               args=self.args)
            tmp_grid.generate_grid()
            self.grid_set.append(tmp_grid)
        return

    def get_user_record_in_attr_group(self, user_record_i,
                                      attr_group: int = None):
        user_record_in_attr_group = []
        for tmp in self.attr_group_list[attr_group]:
            user_record_in_attr_group.append(user_record_i[tmp])
        return user_record_in_attr_group

    def run_ldp(self, user_record):

        gsize_calculator = choose_granularity.ChooseGranularityBeta(attr_group_list=self.attr_group_list,
                                                                         args=self.args, alpha1=self.alpha1, alpha2=self.alpha2)

        self.user_group_ldp_mech_list = []  # intialize for each time to randomize user data
        for j in range(self.group_num):  # initialize LDP mechanism for each attr group
            tmp_grid = self.grid_set[j]  # the i-th Grid
            tmp_domain_size = len(tmp_grid.cell_list)

            grr_error = gsize_calculator.get_grr_error(tmp_domain_size)
            oue_error = gsize_calculator.get_oue_error()

            tmp_ldr = None
            if self.protocol == FOProtocol.ADAPTIVE:
                aux = 3 * np.exp(self.args.epsilon) + 2
                #if tmp_domain_size < aux and grr_error < oue_error:
                if grr_error < oue_error:
                    tmp_ldr = FO.GRR(domain_size=tmp_domain_size,
                                     epsilon=self.args.epsilon,
                                     args=self.args)
                else:
                    tmp_ldr = FO.OUE(domain_size=tmp_domain_size,
                                     epsilon=self.args.epsilon,
                                     args=self.args)
            elif self.protocol == FOProtocol.GRR:
                tmp_ldr = FO.GRR(domain_size=tmp_domain_size,
                                 epsilon=self.args.epsilon,
                                 args=self.args)
            else:
                tmp_ldr = FO.OUE(domain_size=tmp_domain_size,
                                 epsilon=self.args.epsilon,
                                 args=self.args)
            self.user_group_ldp_mech_list.append(tmp_ldr)


        for i in range(self.args.user_num):

            user_count_by_group = math.floor(self.args.user_num / self.group_num)
            if i >= user_count_by_group*self.group_num:
                break
            group_index_of_user = i // user_count_by_group
            j = group_index_of_user

            # to count the user num of each group
            self.user_group_ldp_mech_list[j].group_user_num += 1
            tmp_grid = self.grid_set[j]
            tmp_real_cell_index = 0
            record = user_record[i]
            user_record_in_attr_group_j = self.get_user_record_in_attr_group(record, j)
            tmp_real_cell_index = tmp_grid.get_cell_index_from_attr_value_set(
                user_record_in_attr_group_j)
            tmp_ldp_mechanism = self.user_group_ldp_mech_list[j]
            tmp_ldp_mechanism.operation_perturb(tmp_real_cell_index)



        # update the perturbed_count of each cell
        for j in range(self.group_num):
            tmp_ldp_mechanism = self.user_group_ldp_mech_list[j]
            tmp_ldp_mechanism.operation_aggregate()
            tmp_grid = self.grid_set[j]  # the j-th Grid
            for k in range(len(tmp_grid.cell_list)):
                aux = tmp_ldp_mechanism.aggregated_count[k]
                tmp_grid.cell_list[k].perturbed_count = aux


    def get_attr_group_list(self, selected_attr_list):

        def judge_sub_attr_list_in_attr_group(sub_attr_list, attr_group):
            if len(sub_attr_list) == 1:
                return sub_attr_list == attr_group
            flag = True
            for sub_attr in sub_attr_list:
                if sub_attr not in attr_group:
                    flag = False
                    break
            return flag

        attr_group_index_list = []
        attr_group_list = []
        for grid in self.grid_set:
            # note that here we judge if tmp_Grid.attr_set belongs to selected_attr_list
            if judge_sub_attr_list_in_attr_group(grid.attr_set, selected_attr_list):
                attr_group_index_list.append(grid.index)
                attr_group_list.append(grid.attr_set)

        return attr_group_index_list, attr_group_list

    def hdg_answer_query(self, query):
        t_grid_ans = []
        attr_group_index_list, attr_group_list = self.get_attr_group_list(query.selected_attr_index_list)

        for k in attr_group_index_list:
            grid = self.grid_set[k]
            grid_query_attr_node_list = []
            for attr in grid.attr_set:
                grid_query_attr_node_list.append(query.attr_node_list[attr])
            t_grid_ans.append(grid.answer_query_with_wu_matrix(grid_query_attr_node_list))

        if query.query_dimension == self.group_attr_num:  # answer the 2-way marginal
            tans_weighted_update = t_grid_ans[0]
        else:
            et = EstimateMethod(args=self.args)
            tans_weighted_update = et.weighted_update(query, attr_group_list, t_grid_ans)

        return tans_weighted_update

    def answer_query_list(self, query_list):
        self.weighted_update_answer_list = []
        for query in query_list:
            tans_weighted_update = self.hdg_answer_query(query)
            self.weighted_update_answer_list.append(tans_weighted_update)
        return

    def get_t_a_a(self, sub_attr_value=None, sub_attr=None,
                  relevant_attr_group_list: list = None,
                  c_reci_list=None):

        sum_c_reci_list = sum(c_reci_list)
        sum_t_v_i_a = 0
        for group_index in relevant_attr_group_list:
            t_v_i_a = 0
            grid = self.grid_set[group_index]

            # 1D grids
            if len(grid.attr_set) == 1:
                int_map = self.attr2cell_lens[sub_attr][2]  # g2index2g1_lens_int_index
                left_interval_1_way = int_map[sub_attr_value][0]
                right_interval_1_way = int_map[sub_attr_value][1]
                k = left_interval_1_way
                while k <= right_interval_1_way:
                    try:
                        gcell = grid.cell_list[k]
                    except Exception as exc:
                        print("ERROR {}. k={} list_size={}".format(exc, k, len(grid.cell_list)))
                    t_v_i_a += gcell.consistent_count
                    k += 1
            # 2D grids
            else:
                sub_attr_index_in_grid = grid.attr_set.index(sub_attr)
                for gcell in grid.cell_list:
                    if gcell.cell_pos[sub_attr_index_in_grid] == sub_attr_value:
                        t_v_i_a += gcell.consistent_count
            sum_t_v_i_a += (c_reci_list[group_index] * t_v_i_a)
        if sum_c_reci_list == 0:
            raise Exception("get_t_a_a  self.sum_c_reci_list==0")
        t_a_a = sum_t_v_i_a / sum_c_reci_list
        return t_a_a

    def get_c_w_list(self, sub_attr=None, relevant_attr_group_list: list = None):
        c_reci_list = np.zeros(self.args.group_num)
        for group_index in relevant_attr_group_list:
            grid = self.grid_set[group_index]  # relevant group grid
            if self.g2_list[sub_attr] == 0:
                raise Exception("get_c_w_list  self.g2_list[sub_attr]==0")
            if len(grid.attr_set) == 1:  # 1D grid
                if self.g1_list[sub_attr] // self.g2_list[sub_attr] == 0:
                    print("{} {}".format(self.g1_list[sub_attr], self.g2_list[sub_attr]))
                c_reci_list[group_index] = 1 / (self.g1_list[sub_attr] // self.g2_list[sub_attr])
            else:
                if self.g2_list[sub_attr] == 0:
                    print("{}".format(self.g2_list[sub_attr]))
                c_reci_list[group_index] = 1 / self.g2_list[sub_attr]
        return c_reci_list

    def get_consistency_for_sub_attr(self, sub_attr_index=None):
        try:
            relevant_attr_group_list = []
            for i in range(self.group_num):
                if sub_attr_index in self.attr_group_list[i]:
                    relevant_attr_group_list.append(i)

            sub_attr_domain = range(self.g2_list[sub_attr_index])
            for sub_attr_value in sub_attr_domain:
                c_reci_list = self.get_c_w_list(sub_attr_index, relevant_attr_group_list)
                t_a_a = self.get_t_a_a(sub_attr_value, sub_attr_index, relevant_attr_group_list, c_reci_list)

                for g_index in relevant_attr_group_list:
                    t_v_i_a = 0
                    t_v_i_c_cell_list = []
                    grid = self.grid_set[g_index]
                    if len(grid.attr_set) == 1:
                        int_map = self.attr2cell_lens[sub_attr_index][2]  # g2index2g1_lens_intervals
                        left_interval_1_way = int_map[sub_attr_value][0]
                        right_interval_1_way = int_map[sub_attr_value][1]
                        k = left_interval_1_way
                        while k <= right_interval_1_way:
                            tmp_cell = grid.cell_list[k]
                            t_v_i_c_cell_list.append(k)
                            t_v_i_a += tmp_cell.consistent_count
                            k += 1
                    else:
                        sub_attr_index_in_grid = grid.attr_set.index(sub_attr_index)
                        for k in range(len(grid.cell_list)):
                            cell = grid.cell_list[k]
                            if cell.cell_pos[sub_attr_index_in_grid] == sub_attr_value:
                                t_v_i_c_cell_list.append(k)
                                t_v_i_a += cell.consistent_count

                    for k in t_v_i_c_cell_list:
                        cell = grid.cell_list[k]
                        aux1 = c_reci_list[g_index]
                        aux2 = t_a_a - t_v_i_a
                        cell.consistent_count += aux2 * aux1
        except Exception as ex:
            print("error x: " + ex)
        return

    def overall_consistency(self):
        for i in range(self.args.attr_num):
            self.get_consistency_for_sub_attr(i)
        return

    def get_consistent_grid_set(self):
        for grid in self.grid_set:
            grid.get_consistent_grid()

        self.overall_consistency()
        for i in range(self.args.consistency_iteration_num_max):
            for grid in self.grid_set:
                grid.get_consistent_grid()
            self.overall_consistency()

        # end with the Non-Negativity step
        for tmp_grid in self.grid_set:
            tmp_grid.get_consistent_grid()

    def wu_iteration(self, grid_1d_list=None, grid_2d=None):

        # array_has_nan = np.isnan(grid_2d.wu_matrix)
        # for v in array_has_nan[0]:
        #     if v:
        #         aux = 0

        # update using 1_way
        for grid_1d in grid_1d_list:
            one_way_attr = grid_1d.attr_set[0]
            one_way_attr_index = grid_2d.attr_set.index(one_way_attr)
            for cell in grid_1d.cell_list:  # for each cell of the 1D grid
                lower = cell.left_interval_list[0]
                upper = cell.right_interval_list[0] + 1
                if one_way_attr_index == 0:  # attr in the y-axis

                    # g_shape = grid_2d.wu_matrix.shape
                    # if upper > g_shape[0]:
                    #     aux = 0

                    sub_matrix = grid_2d.wu_matrix[lower:upper, :]
                    tmp_sum = np.sum(sub_matrix)

                    # if math.isnan(tmp_sum):
                    #     aux = 0
                    if tmp_sum == 0:
                        continue  # go to next cell
                    # if math.isnan(cell.consistent_count):
                    #     aux = 0
                    # if cell.consistent_count == 0:
                    #     aux = 0
                    grid_2d.wu_matrix[lower:upper, :] = grid_2d.wu_matrix[lower:upper,
                                                        :] / tmp_sum * cell.consistent_count
                else:  # attr in the x-axis
                    sub_matrix = grid_2d.wu_matrix[:, lower:upper]
                    tmp_sum = np.sum(sub_matrix)

                    # if math.isnan(tmp_sum):
                    #     aux = 0
                    if tmp_sum == 0:
                        continue
                    # if math.isnan(cell.consistent_count):
                    #     aux = 0
                    # if cell.consistent_count == 0:
                    #     aux = 0

                    grid_2d.wu_matrix[:, lower:upper] = grid_2d.wu_matrix[:, lower:upper] / tmp_sum * cell.consistent_count
                # normalization

                # array_has_nan = np.isnan(grid_2d.wu_matrix)
                # for v in array_has_nan[0]:
                #     if v:
                #         aux = 0

                grid_2d.wu_matrix = grid_2d.wu_matrix / np.sum(grid_2d.wu_matrix) * self.args.user_num

        # update using 2_way
        for cell in grid_2d.cell_list:
            x_lower = cell.left_interval_list[0]
            x_upper = cell.right_interval_list[0] + 1
            y_lower = cell.left_interval_list[1]
            y_upper = cell.right_interval_list[1] + 1

            # g_shape = grid_2d.wu_matrix.shape
            # if x_upper > g_shape[0]:
            #     aux = 0
            #
            # if y_upper > g_shape[1]:
            #     aux = 0

            sub_matrix = grid_2d.wu_matrix[x_lower:x_upper, y_lower:y_upper]
            tmp_sum = np.sum(sub_matrix)
            if tmp_sum == 0:
                continue

            # if math.isnan(tmp_sum):
            #     aux = 0

            # if math.isnan(cell.consistent_count):
            #     aux = 0

            grid_2d.wu_matrix[x_lower:x_upper, y_lower:y_upper] = grid_2d.wu_matrix[x_lower:x_upper,
                                                                  y_lower:y_upper] / tmp_sum * cell.consistent_count

            # array_has_nan = np.isnan(grid_2d.wu_matrix)
            # for v in array_has_nan[0]:
            #     if v:
            #         aux = 0

            tmp_sum = np.sum(grid_2d.wu_matrix)

            # if math.isnan(tmp_sum):
            #     aux = 0

            if tmp_sum == 0:
                continue

            # normalization
            grid_2d.wu_matrix = grid_2d.wu_matrix / tmp_sum * self.args.user_num

            # array_has_nan = np.isnan(grid_2d.wu_matrix)
            # for v in array_has_nan[0]:
            #     if v:
            #         aux = 0

        return

    # end wu_iteration()

    def get_wu_for_2_way_group(self):

        for grid in self.grid_set:
            if len(grid.attr_set) == 2:  # 2D grids

                if grid.attr_types_list[0] == AttrType.categorical and grid.attr_types_list[1] == AttrType.categorical:
                    continue

                grid_1_way_list = []
                for grid_1_way in self.grid_set:
                    # select the 1D grids that correspond to the 2D grid
                    if len(grid_1_way.attr_set) == 1 and grid_1_way.attr_set[0] in grid.attr_set:
                        grid_1_way_list.append(grid_1_way)
                domain_x = grid.domain_list[0]
                domain_y = grid.domain_list[1]
                grid.wu_matrix = np.zeros((domain_x, domain_y))
                grid.wu_matrix[:, :] = self.args.user_num / (domain_x * domain_y)

                for i in range(self.args.wu_iteration_num_max):
                    wu_matrix_before = np.copy(grid.wu_matrix)
                    self.wu_iteration(grid_1_way_list, grid)
                    wu_matrix_delta = np.sum(np.abs(grid.wu_matrix - wu_matrix_before))
                    if wu_matrix_delta < 0.1:
                        break
        return
