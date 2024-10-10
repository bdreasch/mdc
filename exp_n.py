import concurrent.futures
import numpy as np
import utility_metric as UM
import generate_query as GenQuery
import random
import TDG
import HDG
import parameter_setting as para
import datetime
from generate_query import AttrType
from frequency_oracle import FOProtocol
from baseline.baseline_HDG import BaselineHDG
from baseline.baseline_TDG import BaselineTDG


kHDG = "HDG"
kHDG_default = "HDG_default"
kHDG_default_not_consistent = "HDG_default_not_consistent"
kTDG_not_consistent = "TDG_default_not_consistent"
kTDG = "TDG"
kB_HDG = "B_HDG"
kB_TDG = "B_TDG"
ADD_NUM_BASELINE = False



# DATASET = "uniform"
# DATASET = "normal"
DATASET = "ipums"
# DATASET = "loan"

DATA_TYPE = "mixed"
# DATA_TYPE = "num"

class ExpResult:
    def __init__(self, maes, stds):
        self.stds = stds
        self.maes = maes

def setup_args(args=None):
    args.attr_num = 6
    args.dimension_query_volume = 0.5
    args.rx = args.dimension_query_volume
    args.ry = args.dimension_query_volume
    args.query_num = 1
    args.query_dimension = 4
    args.selectivity = True


def load_dataset(txt_dataset_path=None):
    user_record = []
    domains_list = []
    read_domain = False
    read_attr_types = True
    attr_types_list = []
    with open(txt_dataset_path, "r") as fr:
        for line in fr:
            line = line.strip()
            line = line.split()
            if read_attr_types:
                attr_types_list_tmp = list(map(str, line))
                for t in attr_types_list_tmp:
                    if t == "c":
                        attr_types_list.append(AttrType.categorical)
                    elif t == "n":
                        attr_types_list.append(AttrType.numerical)
                    else:
                        raise Exception("Invalid attr type in the dataset file")
                read_attr_types = False
                read_domain = True
                continue
            if read_domain:
                domains_list = list(map(int, line))
                read_domain = False
            else:
                user_record.append(list(map(int, line)))
    return user_record, domains_list, attr_types_list


def sys_test(eps_list, algo, phi, repeat, user_record, query_list,
             domains_list, attr_type_list, user_num):
    #print("Executing {} {}".format(algo, phi))
    args = para.generate_args()  # define the parameters
    setup_args(args=args)  # setup the parameters
    args.user_num = user_num
    mae_list = []
    std_list = []
    args.algorithm_name = algo

    for eps in eps_list:
        error_repeat = np.zeros(repeat)
        for r in range(0, repeat):
            args.epsilon = eps
            if args.selectivity:
                args.rx = args.dimension_query_volume
                args.ry = args.dimension_query_volume
            else:
                args.phi_1 = 0.5
                args.phi_2 = 0.5

            if r == 0:
                args.log_g = False
            else:
                args.log_g = False

            if args.algorithm_name == kTDG or args.algorithm_name == kTDG_not_consistent:
                aa = TDG.AGUniformGridOptimal(domain_list=domains_list,
                                              attr_type_list=attr_type_list,
                                              args=args, protocol=FOProtocol.ADAPTIVE,
                                              alpha2=phi)

            elif args.algorithm_name == kHDG:
                aa = HDG.AGUniformGrid12wayOptimal(domain_list=domains_list,
                                                   attr_type_list=attr_type_list,
                                                   args=args, protocol=FOProtocol.ADAPTIVE,
                                                   alpha1=0.7, alpha2=phi, default=False)
            elif args.algorithm_name == kHDG_default:
                aa = HDG.AGUniformGrid12wayOptimal(domain_list=domains_list,
                                                   attr_type_list=attr_type_list,
                                                   args=args, protocol=FOProtocol.ADAPTIVE,
                                                   alpha1=0.7, alpha2=phi, default=True)

            elif args.algorithm_name == kB_HDG:
                args.domain_size = 100
                aa = BaselineHDG(args=args)
            elif args.algorithm_name == kB_TDG:
                args.domain_size = 100
                aa = BaselineTDG(args=args)

            aa.generate_attr_group()

            aa.construct_grid_set()
            aa.run_ldp(user_record)

            aa.get_consistent_grid_set()

            if args.algorithm_name == kHDG or \
                    args.algorithm_name == kHDG_default or \
                    args.algorithm_name == kHDG_default_not_consistent:
                aa.get_wu_for_2_way_group()

            if args.algorithm_name == kB_HDG:
                aa.get_weight_update_for_2_way_group()


            aa.answer_query_list(query_list.query_list)
            um = UM.UtilityMetric(args=args)
            mae_error = um.MAE(query_list.real_answer_list, aa.weighted_update_answer_list)
            error_repeat[r] = mae_error

        mae_list.append(np.mean(error_repeat))
        std_list.append(np.std(error_repeat))


    # else:
    #print("Finished {} {}".format(algo, phi))
    res = ExpResult(mae_list, std_list)
    return res


def read_dataset(dataset, args):

    # list of lists
    user_record, domains_list, attr_types_list = load_dataset(dataset)

    random_seed = 1
    random.seed(random_seed)
    np.random.seed(seed=random_seed)

    query_list = GenQuery.QueryList(
        query_dimension=args.query_dimension,
        attr_num=args.attr_num,
        query_num=args.query_num,
        dimension_query_volume=args.dimension_query_volume,
        domains_list=domains_list,
        attr_types_list=attr_types_list,
        args=args)

    ok = True
    while ok:
        query_list.generate_query_list()
        ok = query_list.generate_real_answer_list(user_record)


    txt_file_path = "exp_n_output/sql_query_" + DATASET + "_lambda=" + str(args.query_dimension) + "_type=" + DATA_TYPE + ".txt"
    with open(txt_file_path, "w") as txt_fr_out:
        query_list.print_query_list(txt_fr_out)

    txt_file_path = "exp_n_output/answers_" + DATASET + "_lambda=" + str(args.query_dimension) + "_type=" + DATA_TYPE + ".txt"
    with open(txt_file_path, "w") as txt_fr_out:
        query_list.print_query_answers(txt_fr_out)

    np.random.shuffle(user_record)
    return user_record, query_list, domains_list, attr_types_list


def run_dataset(dataset, user_num):
    algo_l = []
    phi_l = [0.03]

    algo_l += [kHDG] * len(phi_l)

    # algo_l += [kTDG] * len(phi_l)
    # phi_l += phi_l

    if ADD_NUM_BASELINE:
        algo_l += [kB_TDG]
        phi_l += [0.03]
        algo_l += [kB_HDG]
        phi_l += [0.03]

    eps_l = [1.0]
    times_r = 5

    plot_args = para.generate_args()  # define the parameters
    setup_args(args=plot_args)  # setup the parameters
    plot_args.user_num = user_num
    #print("Repeat: {}\nVolume: {}".format(times_r, plot_args.dimension_query_volume))

    u_ser_record, query_list, domain_list, attr_types_list = read_dataset(dataset=dataset, args=plot_args)

    plot_data = {kHDG_default: [], kHDG: [],  kTDG: [], kB_HDG: [], kB_TDG: []}
    config_list = {kHDG_default: [], kHDG: [],  kTDG: [], kB_HDG: [], kB_TDG: []}
    total_mae_list = {kHDG_default: [], kHDG: [],  kTDG: [], kB_HDG: [], kB_TDG: []}

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        future_to_config = {executor.submit(sys_test, eps_l, algo, phi, times_r,
                                            u_ser_record, query_list,
                                            domain_list, attr_types_list, user_num):
                                [algo, phi] for algo, phi in zip(algo_l, phi_l)}
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                data = future.result()
                maes = data.maes
                stds = data.stds
            except Exception as exc:
                print('%r generated an exception: %s' % (config, exc))
            else:
                plot_data[str(config[0])].append([maes, stds, str(config[0]) + " phi=" + str(config[1])])
                #print("Config: {} Total mae: {:.12f}".format(config, np.sum(maes)))
                total_mae_list[str(config[0])].append(np.sum(maes))
                config_list[str(config[0])].append(str(config[0]) + " " + str(config[1]))
                for e, mae, std in zip(eps_l, maes, stds):
                    #print("eps: {} mae: {:.12f} std: {:.12f}".format(e, mae, std))
                    print("{:.12f}".format(mae))

        for algo, algo_res in plot_data.items():
            with open("exp_n_output/results_" + DATASET + "_" + DATA_TYPE + "_" + str(user_num) + "_"
                      + str(plot_args.dimension_query_volume)
                      + "_" + algo + ".txt", 'w') as f:
                f.write(str(datetime.datetime.now()) + "\n")
                f.write(dataset + "\n\n")
                f.write("N={} lambda={} q_num={} attr_num={} volume={}".format(
                    plot_args.user_num,
                    plot_args.query_dimension,
                    plot_args.query_num,
                    plot_args.attr_num,
                    plot_args.dimension_query_volume))
                f.write('\n')
                f.write("eps: {}".format(eps_l))
                f.write('\n\n')
                for c, m in zip(config_list[algo], total_mae_list[algo]):
                    f.write("config: {} total_mae: {:.12f}".format(c, m))
                    f.write('\n')
                f.write('\n')
                for d_item in algo_res:
                    f.write("config: {}".format(d_item[2]))
                    f.write('\n')
                    for e, mae in zip(eps_l, d_item[0]):
                        f.write("{:.12f}".format(mae))
                        f.write('\n')
                    f.write('\n')

                    for std in d_item[1]:
                        f.write("{:.12f}".format(std))
                        f.write('\n')
                    f.write('\n')

if __name__ == '__main__':

    # uniform, normal, ipums
    datasets = [100000, 158489, 251188,
                398107, 630957, 1000000,
                1584893, 2511886, 3981071,
                6309573, 10000000]


    # loan
    # datasets = [15848, 25118, 39810, 63095, 100000, 158489, 251188, 398107, 630957, 1000000, 1584893]

    for d in datasets:
        path = "test_dataset/" + DATASET + "_n" + str(d) + "_" + DATA_TYPE + ".txt"
        run_dataset(path, d)


