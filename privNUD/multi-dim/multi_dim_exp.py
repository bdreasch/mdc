import numpy as np
import utility_metric as UM
import generate_query as GenQuery
import random
import HDG
import parameter_setting as para

def setup_args(args = None):
    args.algorithm_name = 'HDG'
    args.user_num = 0
    args.attribute_num = 6
    args.domain_size = 3901
    args.epsilon = 1
    args.dimension_query_volume = 0.7
    args.query_num = 100
    args.query_dimension = 3
    args.user_alpha = 0.8


def load_dataset(txt_dataset_path = None):
    user_record = []
    with open(txt_dataset_path, "r") as fr:
        i = 0
        for line in fr:
            line = line.strip()
            line = line.split()
            user_record.append(list(map(int, line)))
            i += 1
    return user_record



def sys_test(eps, repeat_time):

    args = para.generate_args() # define the parameters
    setup_args(args=args)  # setup the parameters
    txt_dataset_path = "./test_dataset/ipums_1m_6a.txt"

    user_record = load_dataset(txt_dataset_path=txt_dataset_path) # read user data
    args.epsilon = eps
    args.user_num = len(user_record)

    # generate range query****************************************************************

    random_seed = 1
    random.seed(random_seed)
    np.random.seed(seed=random_seed)

    range_query = GenQuery.RangeQueryList(query_dimension=args.query_dimension,
                                          attribute_num=args.attribute_num,
                                          query_num=args.query_num,
                                          dimension_query_volume=args.dimension_query_volume, args=args)
    # 读查询集
    query_path = "./test_dataset/ipums100_query.txt"
    query_interval_table = np.loadtxt(query_path, int)

    range_query.generate_range_query_list(query_interval_table)
    range_query.generate_real_answer_list(user_record)

    #  txt_file_path = "test_output/range_query.txt" # print the generated range queries
    txt_file_path = "test_output/query_set_2.txt"  # print the generated range queries
    with open(txt_file_path, "w") as txt_fr_out:
        range_query.print_range_query_list(txt_fr_out)


    MSE_list = []
    for rep in range(repeat_time):
        random_seed = rep
        random.seed(random_seed)
        np.random.seed(seed=random_seed)

        np.random.shuffle(user_record)

        # if args.algorithm_name == 'TDG':
        #     aa = TDG.AG_Uniform_Grid_optimal(args=args)
        # elif args.algorithm_name == "HDG":
        aa = HDG.AG_Uniform_Grid_1_2_way_optimal_2(args=args)

        aa.group_attribute()
        aa.construct_Grid_set()
        aa.get_LDP_Grid_set_divide_user_2(user_record, alpha=args.user_alpha)
        aa.get_consistent_Grid_set_2()

        if args.algorithm_name == 'HDG':
            aa.get_weight_update_for_2_way_group()

        aa.answer_range_query_list(range_query.range_query_list)

        bb = UM.UtilityMetric(args=args)
        MSE = bb.MAE(range_query.real_answer_list, aa.weighted_update_answer_list)
        #MSE = bb.MSE(range_query.real_answer_list, aa.weighted_update_answer_list)
        MSE_list.append(MSE)
    return np.mean(MSE_list)


if __name__ == '__main__':
    eps_list = [0.1, 0.5, 1, 1.5, 2]

    repeat_time = 10
    for eps in eps_list:
        error = sys_test(eps)
        print("{}".format(error))
