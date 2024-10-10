import baseline.b_consistency_method as ConMeth
import math

class GridCell:
    def __init__(self, dimension_num=None, level=0, cell_index=None):
        assert dimension_num is not None
        self.dimension = dimension_num
        self.left_interval_list = [-1 for i in range(dimension_num)]    # x_1, y_1, z_1
        self.right_interval_list = [-1 for i in range(dimension_num)]   # x_2, y_2, z_2
        self.index = cell_index
        self.dimension_index_list = None    # for overall consistency
        self.level = level
        self.next_level_grid = None
        self.real_count = 0
        self.perturbed_count = 0
        self.consistent_count = 0


class UniformGrid:
    def __init__(self, attr_set=None, granularity=None,
                 left_interval_list=None, right_interval_list=None,
                 args=None):
        self.args = args
        self.attr_set = attr_set
        self.dimension = len(attr_set)
        self.index = None
        if granularity is None:
            granularity = self.args.domain_size

        # initialize the intervals
        if left_interval_list is None:
            self.left_interval_list = [0 for i in range(self.dimension)]
        else:
            self.left_interval_list = left_interval_list
        if right_interval_list is None:
            self.right_interval_list = [(self.args.domain_size - 1) for i in range(self.dimension)]
        else:
            self.right_interval_list = right_interval_list

        self.granularity = granularity
        self.cell_length_list = []
        self.cell_list = []
        self.weighted_update_matrix = []

    def get_consistent_grid(self):
        cell_est_list = [0 for i in range(len(self.cell_list))]
        for i in range(len(self.cell_list)):
            cell_est_list[i] = self.cell_list[i].perturbed_count
        consistent_value_list = ConMeth.norm_sub(cell_est_list, user_num=self.args.user_num)

        # set the consistent count to each cell
        for i in range(len(self.cell_list)):
            self.cell_list[i].consistent_count = consistent_value_list[i]
        return

    def get_consistent_grid_iteration(self):
        cell_est_list = [0 for i in range(len(self.cell_list))]
        for i in range(len(self.cell_list)):
            # here is the consistent_count for iteration
            cell_est_list[i] = self.cell_list[i].consistent_count
        consistent_value_list = ConMeth.norm_sub(cell_est_list, user_num= self.args.user_num)
        for i in range(len(self.cell_list)):
            self.cell_list[i].consistent_count = consistent_value_list[i]
        return

    def get_dimension_index_list_from_cell_index(self, cell_index=None):
        dimension_index_list = [0 for i in range(self.dimension)]
        tmp_num = self.dimension - 1
        while tmp_num >= 0:
            dimension_index_list[tmp_num] = cell_index // (self.granularity ** tmp_num)
            cell_index = cell_index % (self.granularity ** tmp_num)
            tmp_num -= 1
        return dimension_index_list

    def get_left_right_interval_from_dimension_index(self, dimension=None,
                                                     dimension_index=None):
        # dimension index denotes the index in the dimension
        li = dimension_index * self.cell_length_list[dimension]
        ri = li + self.cell_length_list[dimension] - 1
        return li, ri

    def generate_grid(self):
        for i in range(self.dimension):
            ri = self.right_interval_list[i]
            li = self.left_interval_list[i]
            if self.granularity == 0:
                raise Exception("Zero div: granularity==0")
            cell_length = math.ceil((ri - li + 1) / self.granularity)
            self.cell_length_list.append(cell_length)

        total_cell_num = self.granularity ** self.dimension
        for i in range(total_cell_num):
            new_cell = GridCell(self.dimension)
            new_cell.index = i
            dimension_index_list = self.get_dimension_index_list_from_cell_index(new_cell.index)
            new_cell.dimension_index_list = dimension_index_list

            for j in range(self.dimension):
                dimension_index = dimension_index_list[j]
                l_interval, r_interval = self.get_left_right_interval_from_dimension_index(j, dimension_index)
                new_cell.left_interval_list[j] = l_interval
                new_cell.right_interval_list[j] = r_interval
            self.cell_list.append(new_cell)
        return

    def add_real_user_record_to_grid(self, attr_value_set: list = None):
        tmp_cell_index = self.get_cell_index_from_attr_value_set(attr_value_set)
        self.cell_list[tmp_cell_index].real_count += 1

    def add_perturbed_user_record_to_grid(self, attr_value_set : list = None):
        tmp_cell_index = self.get_cell_index_from_attr_value_set(attr_value_set)
        self.cell_list[tmp_cell_index].perturbed_count += 1

    def get_cell_index_from_attr_value_set(self, attr_value_set: list = None):
        tmp_dimension_index_list = [-1 for i in range(self.dimension)]
        for i in range(self.dimension):
            if self.cell_length_list[i] == 0:
                raise Exception("Zero div: self.cell_length_list[i]==0")
            tmp_dimension_index_list[i] = attr_value_set[i] // self.cell_length_list[i]
        tmp_cell_index = 0
        for i in range(self.dimension):
            tmp_cell_index += tmp_dimension_index_list[i] * (self.granularity ** i)
        return tmp_cell_index

    def get_answer_range_query_of_cell(self, cell: GridCell = None,
                                       range_query_node_list: list = None,
                                       private_flag=0):
        query_left_interval_list = []
        query_right_interval_list = []
        for i in range(len(range_query_node_list)):
            query_left_interval_list.append(range_query_node_list[i].left_interval)
            query_right_interval_list.append(range_query_node_list[i].right_interval)
        cell_point_counts = 1
        for i in range(len(cell.left_interval_list)):
            cell_length = cell.right_interval_list[i] - cell.left_interval_list[i]
            cell_point_counts *= (cell_length + 1)  # note here 1 must be added

        overlap_point_counts = 1
        for i in range(len(cell.left_interval_list)):
            cell_length = cell.right_interval_list[i] - cell.left_interval_list[i]
            query_length = query_right_interval_list[i] - query_left_interval_list[i]
            min_interval = min(cell.left_interval_list[i], query_left_interval_list[i])
            max_interval = max(cell.right_interval_list[i], query_right_interval_list[i])
            overlap_length = cell_length + query_length - abs(max_interval - min_interval) + 1
            if overlap_length <= 0:
                overlap_point_counts = 0
                break
            else:
                overlap_point_counts *= overlap_length

        tans = overlap_point_counts / cell_point_counts * cell.consistent_count
        return tans

    def get_answer_range_query_of_cell_with_weight_update_matrix(self, cell: GridCell = None,
                                                                 range_query_node_list: list = None):
        query_left_interval_list = []
        query_right_interval_list = []

        for i in range(len(range_query_node_list)):
            query_left_interval_list.append(range_query_node_list[i].left_interval)
            query_right_interval_list.append(range_query_node_list[i].right_interval)

        cell_point_counts = 1
        for i in range(len(cell.left_interval_list)):
            tmp_cell_length = cell.right_interval_list[i] - cell.left_interval_list[i]
            cell_point_counts *= (tmp_cell_length + 1)  # note here 1 must be added

        overlap_point_counts = 1
        for i in range(len(cell.left_interval_list)):
            tmp_cell_length = cell.right_interval_list[i]-cell.left_interval_list[i]
            tmp_query_length = query_right_interval_list[i] - query_left_interval_list[i]
            tmp_min_interval = min(cell.left_interval_list[i], query_left_interval_list[i])
            tmp_max_interval = max(cell.right_interval_list[i], query_right_interval_list[i])

            tmp_overlap_length = tmp_cell_length + tmp_query_length - abs(tmp_max_interval - tmp_min_interval) + 1
            if(tmp_overlap_length <= 0):
                overlap_point_counts = 0
                break
            else:
                overlap_point_counts *= tmp_overlap_length
        tans = 0
        if overlap_point_counts == cell_point_counts:
            tans = cell.consistent_count
        elif overlap_point_counts == 0:
            tans = 0
        else:
            if len(cell.left_interval_list) == 2:
                for i in range(cell.left_interval_list[0], cell.right_interval_list[0] + 1):
                    if query_left_interval_list[0] <= i and i <= query_right_interval_list[0]:
                        for j in range(cell.left_interval_list[1], cell.right_interval_list[1] + 1):
                            if query_left_interval_list[1] <= j and j <= query_right_interval_list[1]:
                                tans += self.weighted_update_matrix[i][j]
            # else:
            #     cell_length = cell.right_interval_list[i] - cell.left_interval_list[i]
            #     tans += cell.consistent_count * overlap_point_counts / cell_length
        return tans


    def answer_range_query(self, range_query_node_list):
        assert len(self.attr_set) == len(range_query_node_list)
        ans = 0
        for i in range(len(self.cell_list)):
            cell = self.cell_list[i]
            ans += self.get_answer_range_query_of_cell(cell, range_query_node_list)
        return ans


    def answer_range_query_with_weight_update_matrix(self, range_query_node_list):
        assert len(self.attr_set) == len(range_query_node_list)
        ans = 0
        for i in range(len(self.cell_list)):
            tmp_cell = self.cell_list[i]
            ans += self.get_answer_range_query_of_cell_with_weight_update_matrix(tmp_cell,
                                                                                 range_query_node_list)
        return ans


