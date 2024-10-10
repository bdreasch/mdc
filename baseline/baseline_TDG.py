import math
import itertools
import baseline.b_grid_generator as GridGen
from baseline.b_estimate_method import EstimateMethod
import baseline.b_frequency_oracle as FO
from baseline.b_choose_granularity import choose_granularity_beta


class BaselineTDG:

    def __init__(self, args=None):
        self.args = args
        self.group_attr_num = 2  # to construct 2-D grids
        self.group_num = 0
        self.attr_group_list = []  # attr_group
        self.grid_set = []   # note that this is different from the HDRtree
        self.output_file_name = None
        self.answer_list = []
        self.weighted_update_answer_list = []
        self.user_group_ldp_mech_list = []  # LDP mechanism for each attr group
        self.granularity = 0
        self.set_granularity()

    def set_granularity(self):
        gran = choose_granularity_beta(args=self.args)
        tmp_g2 = gran.get_2_way_granularity_for_TDG(ep=self.args.epsilon)
        self.granularity = gran.get_rounding_to_pow_2(gran=tmp_g2)
        #print("btdg gran: {}".format(self.granularity))
        self.args.granularity = self.granularity

    # updates the consistent count for each cell that involves the attr_index
    def get_consistency_for_sub_attr(self, attr_index=None):

        def get_t_a_a(sub_attr_value=None, sub_attr=None,
                      relevant_attr_group_list: list = None):
            sum_t_v_i_a = 0
            j = len(relevant_attr_group_list)
            for i in relevant_attr_group_list:
                t_v_i_a = 0
                grid = self.grid_set[i]
                sub_attr_index_in_grid = grid.attr_set.index(sub_attr)
                for cell in grid.cell_list:
                    if cell.dimension_index_list[sub_attr_index_in_grid] == sub_attr_value:
                        t_v_i_a += cell.consistent_count
                sum_t_v_i_a += t_v_i_a

            t_a_a = sum_t_v_i_a / j
            return t_a_a

        relevant_attr_group_list = []
        for i in range(self.group_num):
            if attr_index in self.attr_group_list[i]:
                relevant_attr_group_list.append(i)

        # sub_attr_domain = range(self.args.granularity)  # need to be changed for 3-way attr group
        for sub_attr_value in range(self.args.granularity):
            t_a_a = get_t_a_a(sub_attr_value, attr_index,
                              relevant_attr_group_list)

            # update T_V_i_c
            for i in relevant_attr_group_list:
                grid = self.grid_set[i]
                sub_attr_index_in_grid = grid.attr_set.index(attr_index)
                t_v_i_a = 0
                t_v_i_c_cell_list = []
                for cell_id in range(len(grid.cell_list)):
                    cell = grid.cell_list[cell_id]
                    if cell.dimension_index_list[sub_attr_index_in_grid] == sub_attr_value:
                        t_v_i_c_cell_list.append(cell_id)
                        t_v_i_a += cell.consistent_count

                for k in t_v_i_c_cell_list:
                    cell = grid.cell_list[k]
                    cell.consistent_count = cell.consistent_count + (t_a_a - t_v_i_a) / len(t_v_i_c_cell_list)
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

        # end with Non-Negativity Step
        for grid in self.grid_set:
            grid.get_consistent_grid_iteration()
        return

    def generate_attr_group(self):
        attr_index_list = [i for i in range(self.args.attr_num)]
        attr_combinations = list(itertools.combinations(attr_index_list,
                                                           self.group_attr_num))
        self.group_num = len(attr_combinations)
        self.args.group_num = self.group_num
        self.attr_group_list = attr_combinations

        # for each attr combination(2d)
        for i in range(len(self.attr_group_list)):
            self.attr_group_list[i] = list(self.attr_group_list[i])
        return

    def construct_grid_set(self):

        # for each group, create a grid
        for i in range(self.group_num):
            grid = GridGen.UniformGrid(self.attr_group_list[i],
                                       granularity=self.granularity,
                                       args=self.args)
            grid.index = i
            grid.generate_grid()
            self.grid_set.append(grid)
        return

    def get_user_record_in_attr_group(self, user_record_i,
                                           attr_group: int = None):
        user_record_in_attr_group = []
        for tmp in self.attr_group_list[attr_group]:
            user_record_in_attr_group.append(user_record_i[tmp])
        return user_record_in_attr_group

    def run_ldp(self, user_record):

        # print("TDG is working...")

        self.user_group_ldp_mech_list = []  # initialize for each time to randomize user data

        for j in range(self.group_num):  # initialize LDP mechanism for each attr group
            grid = self.grid_set[j]
            domain_size = len(grid.cell_list)

            ldr = FO.OUE(domain_size=domain_size,
                        epsilon=self.args.epsilon,
                        sampling_factor=self.group_num,
                        args=self.args)
            # tmp_LDR = FreOra.OLH(domain_size=tmp_domain_size, epsilon= self.args.epsilon, sampling_factor=self.group_num, args=self.args)

            self.user_group_ldp_mech_list.append(ldr)

        try:

            for i in range(self.args.user_num):
                user_count_by_group = math.floor(self.args.user_num / self.group_num)

                if i >= user_count_by_group * self.group_num:
                    break

                group_index_of_user = i // user_count_by_group
                j = group_index_of_user

                self.user_group_ldp_mech_list[j].group_user_num += 1
                user_record_in_attr_group_j = self.get_user_record_in_attr_group(user_record[i], j)

                grid = self.grid_set[j]
                real_cell_index = grid.get_cell_index_from_attr_value_set(user_record_in_attr_group_j)
                ldp_mech = self.user_group_ldp_mech_list[j]
                ldp_mech.operation_perturb(real_cell_index)

        except Exception as ex:
            print("run ldp 170 {}".format(ex))


        # update the perturbed_count of each cell
        for j in range(self.group_num):
            ldp_mech = self.user_group_ldp_mech_list[j]
            ldp_mech.operation_aggregate()
            grid = self.grid_set[j]
            for k in range(len(grid.cell_list)):
                grid.cell_list[k].perturbed_count = ldp_mech.aggregated_count[k]

        return

    def judge_sub_attr_list_in_attr_group(self, sub_attr_list, attr_group):
        flag = True
        for sub_attr in sub_attr_list:
            if sub_attr not in attr_group:
                flag = False
                break
        return flag

    def get_answer_range_query_attr_group_list(self, selected_attr_list):
        answer_range_query_attr_group_index_list = []
        answer_range_query_attr_group_list = []
        for grid in self.grid_set:
            # note that here we judge if tmp_grid.attr_set belongs to selected_attr_list
            if self.judge_sub_attr_list_in_attr_group(grid.attr_set, selected_attr_list):
                answer_range_query_attr_group_index_list.append(grid.index)
                answer_range_query_attr_group_list.append(grid.attr_set)

        return answer_range_query_attr_group_index_list, answer_range_query_attr_group_list

    def answer_range_query(self, range_query, private_flag=0):
        t_grid_ans = []
        group_index_list, attr_set_list = \
            self.get_answer_range_query_attr_group_list(range_query.selected_attr_index_list)

        # for each grid that involves any of the selected attributes
        for k in group_index_list:
            grid = self.grid_set[k]
            grid_range_query_att_node_list = []
            for attr in grid.attr_set:
                grid_range_query_att_node_list.append(range_query.attr_node_list[attr])
            aux = grid.answer_range_query(grid_range_query_att_node_list)
            t_grid_ans.append(aux)

        if range_query.query_dimension == self.group_attr_num:  # answer the 1-way2-way marginal
            tans_weighted_update = t_grid_ans[0]
        else:
            em = EstimateMethod(args=self.args)
            tans_weighted_update = em.weighted_update(range_query,
                                                      attr_set_list,
                                                      t_grid_ans)
        return tans_weighted_update

    def answer_query_list(self, range_query_list):
        self.weighted_update_answer_list = []
        for tmp_range_query in range_query_list:
            tans_weighted_update = self.answer_range_query(tmp_range_query)
            self.weighted_update_answer_list.append(tans_weighted_update)
        return
