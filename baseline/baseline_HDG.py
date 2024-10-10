import numpy as np
import math
import itertools

import baseline.b_grid_generator as GridGen
from baseline.b_estimate_method import EstimateMethod
import baseline.b_frequency_oracle as FO
from baseline.b_choose_granularity import choose_granularity_beta


class BaselineHDG:
    def __init__(self, args=None):
        self.args = args
        self.group_attr_num = 2    # to construct 2-D grids
        self.group_num = 0
        self.attr_group_list = []  # attr_group
        self.grid_set = []
        self.answer_list = []
        self.weighted_update_answer_list = []
        self.granularity = None     # granularity g2
        self.granularity_1_way = None   # granularity g1
        self.user_group_ldp_mech_list = []    # LDP mechanism for each attr group
        self.set_granularity_1_2_way()

    def set_granularity_1_2_way(self):
        gran = choose_granularity_beta(args=self.args)
        tmp_g1 = gran.get_1_way_granularity_for_HDG(ep=self.args.epsilon)
        tmp_g2 = gran.get_2_way_granularity_for_HDG(ep=self.args.epsilon)

        # print("eps: {} g1: {} g2: {}".format(self.args.epsilon, tmp_g1,
        #                                       tmp_g2))

        self.granularity_1_way = gran.get_rounding_to_pow_2(gran=tmp_g1)
        self.granularity = gran.get_rounding_to_pow_2(gran=tmp_g2)
        self.args.granularity_1_way = self.granularity_1_way
        self.args.granularity = self.granularity

    def judge_sub_attr_in_attr_group(self, sub_attr=None, attr_group: list = None):
        if sub_attr in attr_group:
            return True
        else:
            return False

    def get_c_w_list(self, sub_attr_value=None, sub_attr=None,
                     relevant_attr_group_list: list = None):
        c_list = np.zeros(self.args.group_num)
        c_reci_list = np.zeros(self.args.group_num)
        for i in relevant_attr_group_list:
            tmp_grid = self.grid_set[i]
            if len(tmp_grid.attr_set) == 1:
                c_list[i] = self.args.granularity_1_way // self.args.granularity
            else:
                c_list[i] = self.args.granularity
            c_reci_list[i] = 1.0 / c_list[i]
        return c_list, c_reci_list

    def get_t_a_a(self, sub_attr_value=None, sub_attr=None,
                  relevant_attr_group_list: list = None,
                  c_reci_list=None):

        sum_c_reci_list = sum(c_reci_list)
        sum_t_v_i_a = 0
        for i in relevant_attr_group_list:
            t_v_i_a = 0
            tmp_grid = self.grid_set[i]
            if len(tmp_grid.attr_set) == 1:
                left_interval_1_way = \
                    sub_attr_value * (self.args.granularity_1_way // self.args.granularity)
                right_interval_1_way = \
                    (sub_attr_value + 1) * (self.args.granularity_1_way // self.args.granularity) - 1
                k = left_interval_1_way
                while k <= right_interval_1_way:
                    try:
                        tmp_cell = tmp_grid.cell_list[k]
                    except Exception as exc:
                        print("ERROR {}. k={} list_size={}".format(exc, k, len(tmp_grid.cell_list)))
                    t_v_i_a += tmp_cell.consistent_count
                    k += 1
            else:
                sub_attr_index_in_grid = tmp_grid.attr_set.index(sub_attr)
                for tmp_cell in tmp_grid.cell_list:
                    if tmp_cell.dimension_index_list[sub_attr_index_in_grid] == sub_attr_value:
                        t_v_i_a += tmp_cell.consistent_count
            sum_t_v_i_a += (c_reci_list[i] * t_v_i_a)
        t_a_a = sum_t_v_i_a / sum_c_reci_list
        return t_a_a

    def get_consistency_for_sub_attr(self, sub_attr=None):
        relevant_attr_group_list = []
        for i in range(self.group_num):
            if self.judge_sub_attr_in_attr_group(sub_attr, self.attr_group_list[i]):
                relevant_attr_group_list.append(i)

        sub_attr_domain = range(self.args.granularity)     # need to be changed for 3-way attr group
        for sub_attr_value in sub_attr_domain:
            c_list, c_reci_list = self.get_c_w_list(sub_attr_value, sub_attr, relevant_attr_group_list)
            t_a_a = self.get_t_a_a(sub_attr_value, sub_attr, relevant_attr_group_list, c_reci_list)

            for i in relevant_attr_group_list: #   update T_V_i_c
                t_v_i_a = 0
                t_v_i_c_cell_list = []
                tmp_grid = self.grid_set[i]
                if len(tmp_grid.attr_set) == 1:
                    left_interval_1_way = sub_attr_value * (self.args.granularity_1_way //
                                                                 self.args.granularity)
                    right_interval_1_way = (sub_attr_value + 1) * (self.args.granularity_1_way //
                                                                        self.args.granularity) - 1
                    k = left_interval_1_way
                    while k <= right_interval_1_way:
                        tmp_cell = tmp_grid.cell_list[k]
                        t_v_i_c_cell_list.append(k)
                        t_v_i_a += tmp_cell.consistent_count
                        k += 1
                else:
                    sub_attr_index_in_grid = tmp_grid.attr_set.index(sub_attr)
                    for k in range(len(tmp_grid.cell_list)):
                        tmp_cell = tmp_grid.cell_list[k]
                        if tmp_cell.dimension_index_list[sub_attr_index_in_grid] == sub_attr_value:
                            t_v_i_c_cell_list.append(k)
                            t_v_i_a += tmp_cell.consistent_count

                for k in t_v_i_c_cell_list:
                    tmp_cell = tmp_grid.cell_list[k]
                    tmp_cell.consistent_count = tmp_cell.consistent_count + (t_a_a - t_v_i_a) * c_reci_list[i]

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
                grid.get_consistent_grid_iteration()
            self.overall_consistency()

        # end with the Non-Negativity step
        for tmp_grid in self.grid_set:
            tmp_grid.get_consistent_grid_iteration()
        return

    #*************consistency end*******************************

    def weighted_update_iteration(self, grid_1_way_list=None,
                                  grid_2_way=None):
        # update using 1_way
        for tmp_grid_1_way in grid_1_way_list:
            tmp_1_way_attr = tmp_grid_1_way.attr_set[0]
            tmp_1_way_attr_index = grid_2_way.attr_set.index(tmp_1_way_attr)
            for i in range(len(tmp_grid_1_way.cell_list)):
                tmp_cell = tmp_grid_1_way.cell_list[i]
                lower_bound = tmp_cell.left_interval_list[0]
                upper_bound = tmp_cell.right_interval_list[0] + 1
                if tmp_1_way_attr_index == 0:
                    tmp_sum = np.sum(grid_2_way.weighted_update_matrix[lower_bound:upper_bound, :])
                    if tmp_sum == 0:
                        continue
                    grid_2_way.weighted_update_matrix[lower_bound:upper_bound, :] = grid_2_way.weighted_update_matrix[lower_bound:upper_bound, :] / tmp_sum * tmp_cell.consistent_count
                else:
                    tmp_sum = np.sum(grid_2_way.weighted_update_matrix[:, lower_bound:upper_bound])
                    if tmp_sum == 0:
                        continue
                    grid_2_way.weighted_update_matrix[:, lower_bound:upper_bound] = grid_2_way.weighted_update_matrix[:, lower_bound:upper_bound] / tmp_sum * tmp_cell.consistent_count
                # normalization
                grid_2_way.weighted_update_matrix = grid_2_way.weighted_update_matrix / np.sum(grid_2_way.weighted_update_matrix) * self.args.user_num
        # update using 2_way
        for tmp_cell in grid_2_way.cell_list:
            x_lower_bound = tmp_cell.left_interval_list[0]
            x_upper_bound = tmp_cell.right_interval_list[0] + 1
            y_lower_bound = tmp_cell.left_interval_list[1]
            y_upper_bound = tmp_cell.right_interval_list[1] + 1
            tmp_sum = np.sum(grid_2_way.weighted_update_matrix[x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound])
            if tmp_sum == 0:
                continue
            grid_2_way.weighted_update_matrix[x_lower_bound:x_upper_bound,
            y_lower_bound:y_upper_bound] = grid_2_way.weighted_update_matrix[x_lower_bound:x_upper_bound,
                                           y_lower_bound:y_upper_bound] / tmp_sum * tmp_cell.consistent_count
            # normalization
            grid_2_way.weighted_update_matrix = grid_2_way.weighted_update_matrix / np.sum(grid_2_way.weighted_update_matrix) * self.args.user_num

    def get_weight_update_for_2_way_group(self):
        for tmp_grid in self.grid_set:
            if len(tmp_grid.attr_set) == 2:
                grid_1_way_list = []
                for tmp_grid_1_way in self.grid_set:
                    if len(tmp_grid_1_way.attr_set) == 1 and tmp_grid_1_way.attr_set[0] in tmp_grid.attr_set:
                        grid_1_way_list.append(tmp_grid_1_way)
                tmp_grid.weighted_update_matrix = np.zeros((self.args.domain_size, self.args.domain_size))
                # initialize
                tmp_grid.weighted_update_matrix[:,:] = self.args.user_num / (self.args.domain_size * self.args.domain_size)

                for i in range(self.args.wu_iteration_num_max):
                    weighted_update_matrix_before = np.copy(tmp_grid.weighted_update_matrix)
                    self.weighted_update_iteration(grid_1_way_list, tmp_grid)
                    weighted_update_matrix_delta = np.sum(np.abs(tmp_grid.weighted_update_matrix - weighted_update_matrix_before))
                    if weighted_update_matrix_delta < 1:
                        break

    def generate_attr_group(self):
        attr_group_list = []
        attr_index_list = [i for i in range(self.args.attr_num)]
        for attr_index in attr_index_list:
            attr_group_list.append((attr_index,))
        attr_group_2_way_list = list(itertools.combinations(attr_index_list, self.group_attr_num))
        for tmp_attr_group_2_way in attr_group_2_way_list:
            attr_group_list.append(tmp_attr_group_2_way)
        self.group_num = len(attr_group_list)
        self.args.group_num = self.group_num
        self.attr_group_list = attr_group_list
        for i in range(len(self.attr_group_list)):
            self.attr_group_list[i] = list(self.attr_group_list[i])
        return

    def construct_grid_set(self):
        for i in range(self.group_num):
            if len(self.attr_group_list[i]) == 1:
                tmp_grid = GridGen.UniformGrid(self.attr_group_list[i],
                                               granularity=self.granularity_1_way,
                                               args=self.args)
            else:
                tmp_grid = GridGen.UniformGrid(self.attr_group_list[i],
                                               granularity=self.granularity,
                                               args=self.args)
            tmp_grid.grid_index = i
            tmp_grid.generate_grid()
            self.grid_set.append(tmp_grid)
        return

    def get_user_record_in_attr_group(self, user_record_i, attr_group: int = None):
        user_record_in_attr_group = []
        for tmp in self.attr_group_list[attr_group]:
            user_record_in_attr_group.append(user_record_i[tmp])
        return user_record_in_attr_group

    def run_ldp(self, user_record):
        # print("HDG is working...")
        self.user_group_ldp_mech_list = []  # intialize for each time to randomize user data
        for j in range(self.group_num):  # initialize LDP mechanism for each attr group
            tmp_grid = self.grid_set[j]   # the i-th Grid
            tmp_domain_size = len(tmp_grid.cell_list)

            tmp_ldr = FO.OUE(domain_size=tmp_domain_size,
                             epsilon=self.args.epsilon,
                             sampling_factor=self.group_num,
                             args=self.args)
            # tmp_LDR = FreOra.OLH(domain_size=tmp_domain_size, epsilon= self.args.epsilon, sampling_factor=self.group_num, args=self.args)
            self.user_group_ldp_mech_list.append(tmp_ldr)

        for i in range(self.args.user_num):
            user_count_by_group = math.floor(self.args.user_num / self.group_num)
            if user_count_by_group == 0:
                raise Exception('Zero division: user_count_by_group == 0')

            if i >= user_count_by_group * self.group_num:
                break

            group_index_of_user = i // user_count_by_group
            j = group_index_of_user

            # to count the user num of each group
            self.user_group_ldp_mech_list[j].group_user_num += 1
            tmp_grid = self.grid_set[j]
            tmp_real_cell_index = 0
            user_record_in_attr_group_j = self.get_user_record_in_attr_group(user_record[i], j)

            tmp_real_cell_index = tmp_grid.get_cell_index_from_attr_value_set(user_record_in_attr_group_j)

            tmp_ldp_mechanism = self.user_group_ldp_mech_list[j]
            tmp_ldp_mechanism.operation_perturb(tmp_real_cell_index)

        # update the perturbed_count of each cell
        for j in range(self.group_num):
            tmp_ldp_mechanism = self.user_group_ldp_mech_list[j]
            tmp_ldp_mechanism.operation_aggregate()
            tmp_grid = self.grid_set[j]  # the j-th Grid
            for k in range(len(tmp_grid.cell_list)):
                tmp_grid.cell_list[k].perturbed_count = tmp_ldp_mechanism.aggregated_count[k]


    def judge_sub_attr_list_in_attr_group(self, sub_attr_list, attr_group):
        if len(sub_attr_list) == 1:
            return sub_attr_list == attr_group
        flag = True
        for sub_attr in sub_attr_list:
            if sub_attr not in attr_group:
                flag = False
                break
        return flag

    def get_answer_range_query_attr_group_list(self, selected_attr_list):
        answer_range_query_attr_group_index_list = []
        answer_range_query_attr_group_list = []
        for tmp_Grid in self.grid_set:
            # note that here we judge if tmp_Grid.attr_set belongs to selected_attr_list
            if self.judge_sub_attr_list_in_attr_group(tmp_Grid.attr_set, selected_attr_list):
                answer_range_query_attr_group_index_list.append(tmp_Grid.grid_index)
                answer_range_query_attr_group_list.append(tmp_Grid.attr_set)

        return answer_range_query_attr_group_index_list, answer_range_query_attr_group_list

    def answer_range_query(self, range_query):
        t_grid_ans = []
        answer_range_query_attr_group_index_list, answer_range_query_attr_group_list = \
        self.get_answer_range_query_attr_group_list(range_query.selected_attr_index_list)

        for k in answer_range_query_attr_group_index_list:
            tmp_grid = self.grid_set[k]
            grid_range_query_attr_node_list = []
            for tmp_attr in tmp_grid.attr_set:
                grid_range_query_attr_node_list.append(range_query.attr_node_list[tmp_attr])
            t_grid_ans.append(tmp_grid.answer_range_query_with_weight_update_matrix(grid_range_query_attr_node_list))

        if range_query.query_dimension == self.group_attr_num: # answer the 2-way marginal
            tans_weighted_update = t_grid_ans[0]
        else:
            et = EstimateMethod(args=self.args)
            tans_weighted_update = et.weighted_update(range_query,
                                                      answer_range_query_attr_group_list,
                                                      t_grid_ans)

        return tans_weighted_update

    def answer_query_list(self, range_query_list):
        self.weighted_update_answer_list = []
        for query in range_query_list:
            tans_weighted_update = self.answer_range_query(query)
            self.weighted_update_answer_list.append(tans_weighted_update)
        return


