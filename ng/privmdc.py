import copy
import numpy as np
import math
import grid_generator as GridGen
import ng.generic_grids as GenericGridsGen
from estimate_method import EstimateMethod
import frequency_oracle as FO
from frequency_oracle import FOProtocol
import itertools
from ng.correlation_identifier import CorrIdentifier
import os
import ng.groups_finder as GF
import choose_granularity
from ng.users_dist_opt import UserDistOptmizer
import bin_packing_v2 as bin_packing
from generate_query import AttrType
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

kOUE = "OUE"
kGRR = "GRR"


class PrivMDC:
    def __init__(self, args, phase_one_dataset_path, domain_list, attr_type_list,
                 protocol=FOProtocol.ADAPTIVE, alpha1=0.7, alpha2=0.03, default=True,
                 grid_weights=None, selectivities=None):
        self.user_mapping = None
        self.args = args
        self.attr_name_map = {f'a{i}': i for i in range(self.args.attr_num)}
        self.grid_weights = grid_weights
        self.group_attr_num = 2  
        self.group_num = 0
        self.domain_list = domain_list  
        self.attr_type_list = attr_type_list  
        self.attr_group_list = []  
        self.grid_set = []
        self.legacy_grid_sets = []
        self.legacy_groups = []
        self.answer_list = []
        self.weighted_update_answer_list = []
        self.default = default
        self.selectivities = selectivities
        self.g1_list = []
        self.g2_list = []
        self.user_group_ldp_mech_list = []  
        self.group_id2grid_size = {}
        self.legacy_group_id2grid_size = {}
        self.group2d_id2mapping_lens = {}  
        self.group1d_id2cell_lens = {}
        self.protocol = protocol
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.group_id2proto = {}
        self.gs_list = {}   
        for i in range(len(domain_list)):
            self.gs_list[i] = [0, 0, 0, 0, 0, 0, 0]

        self.legacy_gs_list = {}  
        for i in range(len(domain_list)):
            self.legacy_gs_list[i] = [0, 0, 0, 0, 0, 0, 0]

        self.attr2cell_lens = {}
        self.legacy_attr2cell_lens = {}
        self.phase_one_dataset_path = phase_one_dataset_path
        self.grids_weights = None


    def old_generate_attr_group(self):
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

        return


    def generate_attr_group(self):
        # adult_cat6 = {'a': False, 'b': True, 'c': True, 'd': True, 'e': False, 'f': False}
        # adult_cat12 = {'a': False, 'b': True, 'c': True, 'd': True, 'e': False, 'f': False,
        #                'g': True, 'h': True, 'i': True, 'j': True, 'k': True, 'l': True}
        #
        # bfive_cat6 = {'a': False, 'b': True, 'c': True, 'd': True, 'e': True, 'f': True}
        # bfive_cat12 = {'a': False, 'b': True, 'c': True, 'd': True, 'e': True, 'f': True,
        #                'g': True, 'h': True, 'i': True, 'j': True, 'k': True, 'l': True}
        #
        # ipums_cat6 = {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False}
        # ipums_cat12 = {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False,
        #                'g': True, 'h': True, 'i': True, 'j': False, 'k': True, 'l': True}
        #
        # loan_cat6 = {'a': False, 'b': True, 'c': True, 'd': True, 'e': False, 'f': False}
        # loan_cat12 = {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False,
        #                'g': True, 'h': True, 'i': True, 'j': False, 'k': True, 'l': True}

        cat_attrs = {}
        # if self.args.dataset == "adult6":
        #     cat_attrs = adult_cat6
        # elif self.args.dataset == "adult12":
        #     cat_attrs = adult_cat12
        # elif self.args.dataset == "ipums6":
        #     cat_attrs = ipums_cat6
        # elif self.args.dataset == "ipums12":
        #     cat_attrs = ipums_cat12
        # elif self.args.dataset == "loan6":
        #     cat_attrs = loan_cat6
        # elif self.args.dataset == "loan12":
        #     cat_attrs = loan_cat12
        # elif self.args.dataset == "bfive6":
        #     cat_attrs = bfive_cat6
        # elif self.args.dataset == "bfive12":
        #     cat_attrs = bfive_cat12

        if self.args.split_ratio > 0:
            pb_amp = math.log2(1 + (self.args.user_num / (self.args.split_ratio * self.args.user_num )) * (math.exp(self.args.dp_e1) - 1))
            corr_file = './test_dataset/gs/gs_{}_nc{}_sr{}_bn{}_t{}_d{}_e{}.txt'.format(self.args.dataset, self.args.n_code,
                                                                                     self.args.split_ratio,
                                                                                    self.args.bn_degree,
                                                                       self.args.data_type,
                                                                       self.args.attr_num, pb_amp)
            if not os.path.exists(corr_file):
                ci = CorrIdentifier(self.phase_one_dataset_path, bn_degree=self.args.bn_degree, cat_attrs=cat_attrs)
                bn = ci.get_bayesian_network(epsilon=self.args.dp_e1)
                #print(bn)
                for c, parents in bn:
                    if len(parents) == 0:
                        self.attr_group_list.append([self.attr_name_map[c]])
                    else:
                        for p in parents:
                            a1 = self.attr_name_map[c]
                            a2 = self.attr_name_map[p]
                            l = [a1, a2]
                            l.sort()
                            self.attr_group_list.append(l)

                for i in range(len(self.attr_type_list)):
                    a_t = self.attr_type_list[i]
                    if a_t == AttrType.numerical:
                        self.attr_group_list.append([i])

                self.group_num = len(self.attr_group_list)
                print(self.group_num)
                # print(self.attr_group_list)
                self.legacy_groups = self.attr_group_list.copy()
                self.args.group_num = self.group_num

                # find grids
                if self.args.kd:
                    self.attr_group_list = GF.find_grids(self.attr_group_list, self.args.attr_num,
                                                         self.args.dataset, self.args.epsilon,
                                                         self.args.bn_degree)
                    print(self.attr_group_list)
                    self.args.group_num = self.group_num = len(self.attr_group_list)

                with open(corr_file, "w") as file:
                    for g in self.attr_group_list:
                        str1 = ""
                        for i in range(len(g)):
                            str1 += str(g[i])

                            if i < len(g) - 1:
                                str1 += " "
                        file.write(str1)
                        file.write("\n")


            else:
                with open(corr_file, "r") as file:
                    for line in file:
                        line = line.strip()
                        line = line.split()
                        group = list(map(int, line))
                        combinations = list(itertools.combinations(group, 2))
                        for c in combinations:
                            self.legacy_groups.append(list(c))
                        if len(group) == 1:
                            self.legacy_groups.append(group)
                        self.attr_group_list.append(group)
                    #print(self.attr_group_list)

                    self.group_num = len(self.attr_group_list)
                    self.args.group_num = self.group_num

                    if not self.args.kd:
                        self.group_num = len(self.legacy_groups)
                        self.args.group_num = self.group_num
                        self.attr_group_list = self.legacy_groups
        else:
            self.old_generate_attr_group()

        return self.attr_group_list

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

    def construct_gs_list(self, attr_type, domain, attr_id, legacy=False):
        gs_cell_len_list = []
        if attr_type == AttrType.numerical:
            for i in range(6):
                if legacy:
                    gs_cell_len_list.append(self.define_initial_cell_list_lens(domain, self.legacy_gs_list[attr_id][i]))
                else:
                    gs_cell_len_list.append(self.define_initial_cell_list_lens(domain, self.gs_list[attr_id][i]))
        elif attr_type == AttrType.categorical:
            g1_cell_len_list = [1] * domain
            g2_cell_len_list = [1] * domain
        else:
            raise Exception("Invalid attr type")
        return gs_cell_len_list

    def setup_all_grid_formatting(self, legacy=False):

        # make the matching
        for attr_id in range(len(self.domain_list)):
            d = self.domain_list[attr_id]
            t = self.attr_type_list[attr_id]

            find = True
            increase_1d = True
            decrease_2d = False

            gs_cell_len_list = self.construct_gs_list(t, d, attr_id, legacy)


            attr2cell_lens = self.attr2cell_lens
            if legacy:
                attr2cell_lens = self.legacy_attr2cell_lens

            if attr_id not in attr2cell_lens:
                attr2cell_lens[attr_id] = []
            attr2cell_lens[attr_id].append([gs_cell_len_list[0], [], {}])

            for z in range(1, len(gs_cell_len_list)):
                if not gs_cell_len_list[z]:
                    if attr_id not in attr2cell_lens:
                        attr2cell_lens[attr_id] = []
                    attr2cell_lens[attr_id].append([])
                    continue

                l3 = []
                for item in gs_cell_len_list[0]:
                    l3.append(int(item))

                l4 = []
                for item in gs_cell_len_list[z]:
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
                    #self.g1_list[attr_id] += 1
                    #self.g2_list[attr_id] -= 1
                    # if increase_1d:
                    #     self.g1_list[attr_id] += 1
                    #     increase_1d = False
                    # else:
                    #     self.g1_list[attr_id] -= 1
                    #     self.g2_list[attr_id] -= 1
                    print(g1_len_list)
                    print(gs_cell_len_list)
                    raise Exception("[setup_all_grid_formatting] Need to change gs_list content")

                if attr_id not in attr2cell_lens:
                    attr2cell_lens[attr_id] = []

                attr2cell_lens[attr_id].append([g1_len_list, l4, g2index2g1_lens_int_index])
        #print(self.gs_list)
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

    def calculate_inital_grid_kd_size(self, index, legacy=False):

        group = None

        group = self.legacy_groups[index]
        self.legacy_group_id2grid_size[index] = self.group_id2grid_size[index]
        size_per_dimension = self.gs_list
        for i in range(len(size_per_dimension)):
            a = group[i]
            d = self.domain_list[a]
            if size_per_dimension[i] > d:
                size_per_dimension[i] = d

        for i in range(len(group)):
            if legacy:
                self.legacy_gs_list[group[i]][len(group)-1] = size_per_dimension[i]
            else:
                self.gs_list[group[i]][len(group)-1] = size_per_dimension[i]
        return

    def construct_grid_set(self):

        for i in range(len(self.domain_list)):
            self.calculate_inital_grid_1d_size(i)

        for i in range(self.group_num):
            group = self.attr_group_list[i]
            if len(group) > 1:
                self.calculate_inital_grid_kd_size(i)

        self.setup_all_grid_formatting()

        tmp_grid = None
        for i in range(self.group_num):
            group = self.attr_group_list[i]
            group_domain = []
            for attr in group:
                group_domain.append(self.domain_list[attr])
            if len(self.attr_group_list[i]) == 1:  # 1D grid

                valid_list = 0
                for k in range(len(self.attr2cell_lens[attr])):
                    if self.attr2cell_lens[attr][k]:
                        valid_list = k
                        break
                d1_len_list = self.attr2cell_lens[attr][valid_list][0]
                grid_attr_types = [self.attr_type_list[group[0]]]
                gran_list = [self.gs_list[group[0]][0]]
                tmp_grid = GridGen.UniformGrid(index=i,
                                               attr_set=group,
                                               domain_list=group_domain,
                                               attr_types_list=grid_attr_types,
                                               grid_size=gran_list,
                                               cell_length_d1_list=d1_len_list,
                                               args=self.args)
            else:
                type_dims = []
                kd_len_list = []
                gran_list = []
                for g in group:
                    type_dims.append(self.attr_type_list[g])
                    # valid_list = 0
                    # for k in range(len(self.attr2cell_lens[g])):
                    #     if self.attr2cell_lens[g][k]:
                    #         valid_list = k
                    #         break

                    kd_len_list.append(self.attr2cell_lens[g][len(group)-1][1])
                    gran_list.append(self.gs_list[g][len(group)-1])

                tmp_grid = GenericGridsGen.GenericUniformGrid(index=i,
                                               attr_set=group,
                                               domain_list=group_domain,
                                               attr_types_list=type_dims,
                                               grid_size=gran_list,
                                               kd_cell_lens_list=kd_len_list,
                                               args=self.args)
            tmp_grid.generate_grid()
            self.grid_set.append(tmp_grid)

        if self.args.split_ratio > 0:
            for i in range(len(self.domain_list)):
                self.calculate_inital_grid_1d_size(i, legacy=True)

            for i in range(len(self.legacy_groups)):
                group = self.legacy_groups[i]
                if len(group) > 1:
                    self.calculate_inital_grid_kd_size(i, legacy=True)

            self.setup_all_grid_formatting(legacy=True)


            tmp_grid = None
            for i in range(len(self.legacy_groups)):
                group = self.legacy_groups[i]
                group_domain = []
                for attr in group:
                    group_domain.append(self.domain_list[attr])
                if len(self.legacy_groups[i]) == 1:  # 1D grid

                    valid_list = 0
                    for k in range(len(self.legacy_attr2cell_lens[attr])):
                        if self.legacy_attr2cell_lens[attr][k]:
                            valid_list = k
                    d1_len_list = self.legacy_attr2cell_lens[attr][valid_list][0]

                    grid_attr_types = [self.attr_type_list[group[0]]]
                    gran_list = [self.legacy_gs_list[group[0]][0]]
                    tmp_grid = GridGen.UniformGrid(index=i,
                                                   attr_set=group,
                                                   domain_list=group_domain,
                                                   attr_types_list=grid_attr_types,
                                                   grid_size=gran_list,
                                                   cell_length_d1_list=d1_len_list,
                                                   args=self.args)
                else:
                    type_dims = []
                    kd_len_list = []
                    gran_list = []
                    for g in group:
                        type_dims.append(self.attr_type_list[g])

                        # valid_list = 0
                        # for k in range(len(self.attr2cell_lens[g])):
                        #     if self.attr2cell_lens[g][k]:
                        #         valid_list = k
                        #         break
                        #
                        # kd_len_list.append(self.attr2cell_lens[g][valid_list][1])
                        kd_len_list.append(self.legacy_attr2cell_lens[g][len(group)-1][1])
                        gran_list.append(self.legacy_gs_list[g][len(group)-1])

                    tmp_grid = GenericGridsGen.GenericUniformGrid(index=i,
                                                   attr_set=group,
                                                   domain_list=group_domain,
                                                   attr_types_list=type_dims,
                                                   grid_size=gran_list,
                                                   kd_cell_lens_list=kd_len_list,
                                                   args=self.args)
                tmp_grid.generate_grid()
                self.legacy_grid_sets.append(tmp_grid)

        return

    def get_user_record_in_attr_group(self, user_record_i,
                                      attr_group: int = None):
        user_record_in_attr_group = []
        for tmp in self.attr_group_list[attr_group]:
            user_record_in_attr_group.append(user_record_i[tmp])
        return user_record_in_attr_group

    def define_user_mapping(self):
        #print("start user mapping")

        def user_map(user_dist):
            user_m = {}
            for i in range(self.group_num):
                user_m[i] = []

            cursor = 0
            for j in range(len(user_dist)):
                for i in range(cursor, cursor+user_dist[j]):
                    user_m[j].append(i)
                cursor += user_dist[j]
            return user_m

        if not self.args.optimize_udist:
            user_dist = [self.args.user_num//self.group_num] * self.group_num
            diff = self.args.user_num - sum(user_dist)
            index = -1
            while diff > 0:
                user_dist[index] += 1
                index += -1
                if index < -self.args.attr_num:
                    index = -1
                diff = self.args.user_num - sum(user_dist)
            self.user_mapping = user_map(user_dist)
            return


        rs_list = []
        for i in range(len(self.attr_group_list)):
            group = self.attr_group_list[i]
            g_list = []
            for g in group:
                g_list.append(self.selectivities[g])
            rs_list.append(g_list)

        ls_list = []
        for i in range(len(self.attr_group_list)):
            lss = []
            group = self.attr_group_list[i]
            for g in group:
                lss.append(self.gs_list[g][len(group)-1])
            ls_list.append(lss)

        user_dist = []
        dist_file = './cache/dist_ds_{}_nc{}_sr{}_d{}_e{}_bn{}.txt'.format(self.args.dataset, self.args.n_code,
                                                                                self.args.split_ratio,
                                                                                self.args.attr_num,
                                                                                self.args.epsilon, self.args.bn_degree)

        skip = True

        if not os.path.exists(dist_file):
            udo = UserDistOptmizer(rs_list=rs_list, ls_list=ls_list, grids_weights=self.grids_weights, args=self.args)
            user_dist = udo.find_user_dist()
        else:
            gd_sum = sum(self.grids_weights)
            weights = [float(i)/gd_sum for i in self.grids_weights]
            for w in weights:
                user_dist.append(int(w*self.args.user_num))
            diff = sum(user_dist) - self.args.user_num
            if diff < 0:
                while diff < 0:
                    for j in range(len(user_dist)):
                        if user_dist[j] > 0:
                            user_dist[j] += 1
                            diff += 1
                            if diff == 0:
                                break
            elif diff > 0:
                while diff > 0:
                    for j in range(len(user_dist)):
                        if user_dist[j] > 0:
                            user_dist[j] -= 1
                            diff -= 1
                            if diff == 0:
                                break

        diff = sum(user_dist) - self.args.user_num
        with open(dist_file, "w") as file:
            for i in range(len(user_dist)):
                str1 = str(user_dist[i])
                file.write(str1)
                file.write("\n")

        self.user_mapping = user_map(user_dist)
        #print("end user mapping")
        return

    def calculate_grids_weights(self, query_list):
        if not self.args.optimize_udist:
            return np.ones((len(self.attr_group_list),), dtype=int)
        self.grids_weights = np.zeros((len(self.attr_group_list),), dtype=int)
        for query in query_list:
            combinations = list(itertools.combinations(query.selected_attr_index_list, 2))
            for c in combinations:
                pair = list(c)

                g_index = self.attr_group_list.index([pair[0]])
                self.grids_weights[g_index] += 1
                g_index = self.attr_group_list.index([pair[1]])
                self.grids_weights[g_index] += 1

                for g in self.attr_group_list:
                    if set(pair).issubset(g):
                        g_index = self.attr_group_list.index(g)
                        self.grids_weights[g_index] += 1
                        break
        return


    def run_ldp(self, user_record):

        self.user_group_ldp_mech_list = [] 
        for j in range(self.group_num):  
            tmp_grid = self.grid_set[j]  
            tmp_domain_size = len(tmp_grid.cell_list)

            if tmp_domain_size > 3*math.exp(self.args.epsilon) + 2:
                tmp_ldr = FO.OUE(domain_size=tmp_domain_size,
                                 epsilon=self.args.epsilon,
                                 args=self.args)
            else:
                tmp_ldr = FO.GRR(domain_size=tmp_domain_size,
                                 epsilon=self.args.epsilon,
                                 args=self.args)

            self.user_group_ldp_mech_list.append(tmp_ldr)



        for group_index_of_user, users in self.user_mapping.items():
            for i in users:
                # user_count_by_group = math.floor(self.args.user_num / self.group_num)
                # if i >= user_count_by_group * self.group_num:
                #     break
                # group_index_of_user = i // user_count_by_group
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



        for j in range(self.group_num):
            tmp_ldp_mechanism = self.user_group_ldp_mech_list[j]
            if tmp_ldp_mechanism.group_user_num > 0:
                tmp_ldp_mechanism.operation_aggregate()
                tmp_grid = self.grid_set[j]  # the j-th Grid
                for k in range(len(tmp_grid.cell_list)):
                    aux = tmp_ldp_mechanism.aggregated_count[k]
                    tmp_grid.cell_list[k].perturbed_count = aux
            else:
                tmp_grid = self.grid_set[j]
                for k in range(len(tmp_grid.cell_list)):
                    tmp_grid.cell_list[k].perturbed_count = 0


        # for j in range(self.group_num):
        #     print(self.user_group_ldp_mech_list[j].group_user_num)
        return


    def get_attr_group_list(self, selected_attr_list):

        def judge_sub_attr_list_in_attr_group(sub_attr_list, attr_group):
            # if len(sub_attr_list) == 1:
            #     return False
            flag = True
            for sub_attr in sub_attr_list:
                if sub_attr not in attr_group:
                    flag = False
                    break
            return flag

        attr_group_index_list = []
        attr_group_list = []
        for grid in self.grid_set:

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

            try:
                t_grid_ans.append(grid.answer_query_with_wu_matrix(grid_query_attr_node_list))
            except Exception as exc:
                print('hdg_answer_query: %s' % (exc))

        if query.query_dimension == self.group_attr_num:
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
                int_map = self.attr2cell_lens[sub_attr][1][2]  # g2index2g1_lens_int_index
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
        t_a_a = sum_t_v_i_a / sum_c_reci_list
        return t_a_a

    def get_c_w_list(self, sub_attr=None, relevant_attr_group_list: list = None):
        c_reci_list = np.zeros(self.args.group_num)
        for group_index in relevant_attr_group_list:
            grid = self.grid_set[group_index]  # relevant group grid

            if self.args.smart_post:
                if len(grid.attr_set) == 1:  # 1D grid
                    bla = len(self.user_mapping[group_index])
                    c_reci_list[group_index] = (len(self.user_mapping[group_index]) / self.args.user_num) / \
                                               (self.gs_list[sub_attr][0] // self.gs_list[sub_attr][1])
                else:
                    c_reci_list[group_index] = (len(self.user_mapping[group_index]) / self.args.user_num) / \
                                               self.gs_list[sub_attr][1]
            else:
                if len(grid.attr_set) == 1:  # 1D grid
                    c_reci_list[group_index] = 1 / (self.gs_list[sub_attr][0] // self.gs_list[sub_attr][1])
                else:
                    c_reci_list[group_index] = 1 / self.gs_list[sub_attr][1]

        return c_reci_list

    def get_consistency_for_sub_attr(self, sub_attr_index=None):

        try:
            relevant_attr_group_list = []
            for i in range(self.group_num):
                if sub_attr_index in self.attr_group_list[i]:
                    relevant_attr_group_list.append(i)

            sub_attr_domain = range(self.gs_list[sub_attr_index][1]) #sub_attr_domain = range(self.g2_list[sub_attr_index])
            for sub_attr_value in sub_attr_domain:
                c_reci_list = self.get_c_w_list(sub_attr_index, relevant_attr_group_list)
                t_a_a = self.get_t_a_a(sub_attr_value, sub_attr_index, relevant_attr_group_list, c_reci_list)

                for g_index in relevant_attr_group_list:
                    t_v_i_a = 0
                    t_v_i_c_cell_list = []
                    grid = self.grid_set[g_index]
                    if len(grid.attr_set) == 1:
                        int_map = self.attr2cell_lens[sub_attr_index][1][2]  # g2index2g1_lens_intervals
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

        if self.args.consist:
            self.overall_consistency()
            for i in range(self.args.consistency_iteration_num_max):
                for grid in self.grid_set:
                    grid.get_consistent_grid()
                self.overall_consistency()

            # end with the Non-Negativity step
            for tmp_grid in self.grid_set:
                tmp_grid.get_consistent_grid()

    def wu_iteration(self, grid_1d_list=None, grid_2d=None):

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
                    grid_2d.wu_matrix[lower:upper, :] = grid_2d.wu_matrix[lower:upper,:] / tmp_sum * cell.consistent_count
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

                    grid_2d.wu_matrix[:, lower:upper] = grid_2d.wu_matrix[:,
                                                        lower:upper] / tmp_sum * cell.consistent_count
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

    def extract_grids(self):

        if self.args.split_ratio > 0:
            def copy_cell(lg_cell, n_cell):
                lg_cell.dimension = n_cell.dimension
                lg_cell.left_interval_list = n_cell.left_interval_list.copy()
                lg_cell.right_interval_list = n_cell.right_interval_list.copy()
                lg_cell.index = n_cell.index
                lg_cell.cell_pos = n_cell.cell_pos
                lg_cell.level = n_cell.level
                lg_cell.next_level_grid = n_cell.next_level_grid
                lg_cell.real_count = n_cell.real_count
                lg_cell.perturbed_count = n_cell.perturbed_count
                lg_cell.consistent_count = n_cell.consistent_count

            for legacy_grid in self.legacy_grid_sets:
                for ngrid in self.grid_set:
                    if ngrid.attr_set == legacy_grid.attr_set:
                        for lg_cell, n_cell in zip(legacy_grid.cell_list, ngrid.cell_list):
                            #copy_cell(lg_cell, n_cell)
                            lg_cell.perturbed_count = n_cell.perturbed_count

            for ngrid in self.grid_set:
                if ngrid.dimension > 2:
                    combs = itertools.combinations(ngrid.attr_set, 2)
                    for pair_grid in combs:
                        for lg in self.legacy_grid_sets:
                            if lg.attr_set == list(pair_grid):
                                ng_attr_indexes = [ngrid.attr_set.index(pair_grid[0]), ngrid.attr_set.index(pair_grid[1])]
                                #attr_sizes = [self.g2_list[ng_attr_indexes[0]], self.g2_list[ng_attr_indexes[1]]]


                                xs_n = ngrid.grid_size[ng_attr_indexes[0]]
                                ys_n = ngrid.grid_size[ng_attr_indexes[1]]
                                sub_matrix_array = np.zeros([xs_n, ys_n])
                                for ncell in ngrid.cell_list:
                                    pair_index = tuple([ncell.cell_pos[ng_attr_indexes[0]], ncell.cell_pos[ng_attr_indexes[1]]])
                                    sub_matrix_array[pair_index] += ncell.perturbed_count
                                    # [0,0] += [0,0, x, y]

                                xs_lg = lg.grid_size[0]
                                ys_lg = lg.grid_size[1]
                                fx = int(xs_lg / xs_n)
                                fy = int(ys_lg / ys_n)

                                if (xs_lg / xs_n) - fx != 0:
                                    print("x_div problem {}".format(xs_lg / xs_n))
                                fy = int(ys_lg / ys_n)
                                if (ys_lg / ys_n) - fy != 0:
                                    print("y_div problem {}".format(ys_lg / ys_n))

                                upscaled = np.kron(sub_matrix_array, np.ones((fx, fy)))
                                upscaled = upscaled / (fx*fy)

                                for lgc in lg.cell_list:
                                    lgc.perturbed_count = upscaled[tuple(lgc.cell_pos)]





            aux_gs_list = self.gs_list.copy()
            self.gs_list = self.legacy_gs_list.copy()
            self.legacy_gs_list = aux_gs_list.copy()

            aux_grids = self.grid_set.copy()
            self.grid_set = self.legacy_grid_sets.copy()
            self.legacy_grid_sets = aux_grids.copy()
            self.group_num = self.args.group_num = len(self.grid_set)
            aux_groups = self.attr_group_list.copy()
            self.attr_group_list = self.legacy_groups.copy()
            self.legacy_groups = aux_groups.copy()

            aux_legacy_attr2cell_lens = self.attr2cell_lens.copy()
            self.attr2cell_lens = self.legacy_attr2cell_lens.copy()
            self.legacy_attr2cell_lens = aux_legacy_attr2cell_lens.copy()

            aux_legacy_group_id2grid_size = self.group_id2grid_size.copy()
            self.group_id2grid_size = self.legacy_group_id2grid_size.copy()
            self.legacy_group_id2grid_size = aux_legacy_group_id2grid_size.copy()

