import concurrent.futures
import csv
import os
import numpy as np
import pandas as pd
import plot
import utility_metric as UM
import generate_query as GenQuery
#import query_gen_ng as GenQueryNG
import random
import felip
import parameter_setting as para
import datetime
from generate_query import AttrType
from frequency_oracle import FOProtocol
from baseline.baseline_HDG import BaselineHDG
from baseline.baseline_TDG import BaselineTDG
from ng.privmdc import PrivMDC
import warnings
from random import shuffle
import time
from multiprocessing import Lock
from privNUD.privnud import PrivNud
import file_helper as fh
import copy

warnings.filterwarnings("ignore")
kFELIP = "FELIP"
kFELIP_default = "HDG_default"
kFELIP_default_not_consistent = "HDG_default_not_consistent"
kTDG_not_consistent = "TDG_default_not_consistent"
kB_HDG = "HDG"
kB_TDG = "TDG"
kPrivMDC = "PrivMDC"
kHBS = "HBS"
kPrivNud = "PrivNUD"
ADD_NUM_BASELINE = False
# DATA_TYPE = "mixed"
DATA_TYPE = "num"


# DATASET = "ipums12_s1"
#DATASET = "ipums12_s2"
# DATASET = "ipums12_s3"
# DATASET = "ipums12_s4"
# DATASET = "ipums12_s5"

# USER_NUM = 907895

# DATASET = "ipums12b_s1"
# DATASET = "ipums12b_s2"
# DATASET = "ipums12b_s3"
# DATASET = "ipums12b_s4"
# DATASET = "ipums12b_s5"

# DATASET = "ipums12b_w1"


# USER_NUM = 907895
# DATASET = "ipums12b_bm"
# DATASET = "ipums12_ndp"
# TXT_DATASET_PATH = "test_dataset/ipums12b.txt"


# DATASET = "bfive12b_s1"
#DATASET = "bfive12b_s2"
# DATASET = "bfive12b_s3"
# DATASET = "bfive12b_s4"
#DATASET = "bfive12b_s5"


# USER_NUM = 694886
# DATASET = "bfive12b_w4"
# DATASET = "bfive12b_exp_wla"
# DATASET = "bfive12_ndp"
# TXT_DATASET_PATH = "test_dataset/bfive12b.txt"
# 0.9 s with 5lamba


# DATASET = "adult12b_s1"
# DATASET = "adult12b_s2"
# DATASET = "adult12b_s3"
# DATASET = "adult12b_s4"
# DATASET = "adult12b_s5"

# ATTRS = 12
# USER_NUM = 32560
# DATASET = "adult12"
# DATASET = "adult12_w2"
# DATASET = "adult12b_w2"
# TXT_DATASET_PATH = "test_dataset/adult12_exp_dps.txt"
# DATASET = "adult12_ndp"

# DATASET = "adult15b"
# TXT_DATASET_PATH = "test_dataset/adult15b.txt"

#DATASET = "loan12b"
# DATASET = "loan12b_w3"
# DATASET = "loan12_ndp"

ATTRS = 24
USER_NUM = 2260701
DATASET = "loan150b"
TXT_DATASET_PATH = "test_dataset/loan150b.txt"
# DATASET = "loan96b"
# TXT_DATASET_PATH = "test_dataset/loan96b.txt"
# DATASET = "loan48b"
# TXT_DATASET_PATH = "test_dataset/loan48b.txt"
# DATASET = "loan24b"
# TXT_DATASET_PATH = "test_dataset/loan24b.txt"
# DATASET = "loan12b_wal"
# TXT_DATASET_PATH = "test_dataset/loan12b.txt"


# ATTRS = 24
# USER_NUM = 694886
# DATASET = "bfive104b"
# TXT_DATASET_PATH = "test_dataset/bfive104b.txt"
# DATASET = "bfive96b"
# TXT_DATASET_PATH = "test_dataset/bfive96b.txt"
# DATASET = "bfive48b"
# TXT_DATASET_PATH = "test_dataset/bfive48b.txt"
# DATASET = "bfive24b"
# TXT_DATASET_PATH = "test_dataset/bfive24b.txt"


#DATASET = "bfive_c3"
#DATASET = "bfive_c4"
#DATASET = "bfive_c5"
# DATASET = "bfive_c6"
# TXT_DATASET_PATH = "test_dataset/bfive_c6.txt"
# USER_NUM = 694886


# USER_NUM = 990311
# TXT_DATASET_PATH = "test_dataset/ipums130b.txt"
# DATASET = "ipums130b"
# ATTRS = 12
# TXT_DATASET_PATH = "test_dataset/ipums96b.txt"
# DATASET = "ipums96b"
# TXT_DATASET_PATH = "test_dataset/ipums48b.txt"
# DATASET = "ipums48b"
# TXT_DATASET_PATH = "test_dataset/ipums24b.txt"
# DATASET = "ipums24b"
# TXT_DATASET_PATH = "test_dataset/ipums12b.txt"
# DATASET = "ipums12b_bm"



kTimeCount = False
kSelectivities = [0.5, 0.4, 0.7, 0.5, 0.7, 0.8, 0.9, 0.7, 0.7, 0.9, 0.7, 0.5]




df_lock = Lock()

class ExpResult:
    def __init__(self, maes, stds, percentile25_list, percentile75_list, exp_dataframe):
        self.stds = stds
        self.maes = maes
        self.percentile25_list = percentile25_list
        self.percentile75_list = percentile75_list
        self.exp_dataframe = exp_dataframe

def setup_args(args=None):
    args.user_num = USER_NUM
    args.n_code = "1m"
    args.attr_num = ATTRS

    args.dimension_query_volume = 0.9
    args.rx = args.dimension_query_volume
    args.ry = args.dimension_query_volume
    args.query_num = 100
    args.query_dimension = 5
    args.selectivity = True
    args.split_ratio = 0.01
    args.bn_degree = 3
    args.dataset = DATASET
    args.data_type = DATA_TYPE
    args.smart_post = False
    args.kd = True
    args.consist = True
    args.optimize_udist = True


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
                        raise Exception("load_dataset: Invalid attr type in the dataset file")
                read_attr_types = False
                read_domain = True
                continue
            if read_domain:
                domains_list = list(map(int, line))
                read_domain = False
            else:
                user_record.append(list(map(int, line)))

    return user_record, domains_list, attr_types_list


def generate_phase_one_csv(user_record, args, r, force=True):
    phase_one_csv_path = './test_dataset/tmp/{}_n{}_t{}_r{}_d{}_r{}.csv'.format(DATASET, args.user_num, DATA_TYPE,
                                                                        args.split_ratio, args.attr_num, r)
    if not os.path.exists(phase_one_csv_path) or force:
        with open(phase_one_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)

            #decimal_a = 97
            #header = []
            #for i in range(args.attr_num):
            #    header.append(chr(decimal_a))
            #    decimal_a += 1
            header = [f'a{i}' for i in range(args.attr_num)]
            writer.writerow(header)
            for ur in user_record:
                writer.writerow(ur)
    return phase_one_csv_path


def sys_test(eps, algo, repeat, user_record, query_list,
             domains_list, attr_type_list):
    phi = 0.03
    progress = 0
    print("Executing {} {}".format(algo, eps))
    args = para.generate_args()  # define the parameters
    setup_args(args=args)  # setup the parameters

    dp_query_list = copy.deepcopy(query_list)
    mae_list = []
    std_list = []
    percentile25_list = []
    percentile75_list = []
    #exp_dataframe = pd.DataFrame()
    exp_log_list = []
    args.algorithm_name = algo
    ds_name = "invalid"

    error_repeat = np.zeros(repeat)
    for r in range(0, repeat):
        ldp_records = user_record
        args.user_num = USER_NUM
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

        if args.algorithm_name == kFELIP:
            aa = felip.AGUniformGrid12wayOptimal(domain_list=domains_list,
                                               attr_type_list=attr_type_list,
                                               args=args, protocol=FOProtocol.ADAPTIVE,
                                               alpha1=0.7, alpha2=phi, default=False)
        elif args.algorithm_name == kHBS:
            aa = HBS(domain_list=domains_list, attr_type_list=attr_type_list,
                                               args=args, protocol=FOProtocol.ADAPTIVE,
                                               alpha1=0.7, alpha2=phi, default=False)

        elif args.algorithm_name == kPrivNud:
            if "adult" in DATASET or "ipums12b" in DATASET or "loan" in DATASET or "ipums" in DATASET or "bfive" in DATASET:
                args.domain_size = 64

            args.user_alpha = 0.8
            aa = PrivNud(args=args)

        elif args.algorithm_name == kPrivMDC:
            #np.random.seed(1)
            if args.split_ratio > 0:
                rn_array = np.random.choice(args.user_num-1, size=int(args.user_num * args.split_ratio), replace=False)
                phase_one_dataset = []
                phase_two_dataset = []
                for idx in range(args.user_num):
                    if idx in rn_array:
                        phase_one_dataset.append(user_record[idx])
                        phase_two_dataset.append(user_record[idx])
                    else:
                        phase_two_dataset.append(user_record[idx])

                ds_name = generate_phase_one_csv(phase_one_dataset, args, r=r)
                ok = True
                while ok:
                    ok = dp_query_list.generate_real_answer_list(phase_one_dataset)
                    if ok:
                        args.query_num -= 1
                        query_list.query_num = args.query_num
                        if args.query_num == 0:
                            raise Exception("DP ZERO NUMBER OF QUERIES")

                args.user_num = len(phase_two_dataset)
                aa = PrivMDC(args, phase_one_dataset_path=ds_name, domain_list=domains_list,
                             attr_type_list=attr_type_list, selectivities=kSelectivities)

                ldp_records = phase_two_dataset
            else:
                aa = PrivMDC(args, phase_one_dataset_path="", domain_list=domains_list,
                             attr_type_list=attr_type_list, selectivities=kSelectivities)
                ldp_records = user_record
        elif args.algorithm_name == kB_HDG:
            if "adult" in DATASET or "ipums12b" in DATASET or "loan" in DATASET or "ipums" in DATASET or "bfive" in DATASET:
                args.domain_size = 64

            aa = BaselineHDG(args=args)

        allstart = time.time()
        all_groups = aa.generate_attr_group()
        end = time.time()
        if kTimeCount:
            print("generate_attr_group: {}".format(end - allstart))

        if args.algorithm_name == kPrivMDC:
            aa.calculate_grids_weights(query_list.query_list)

        start = time.time()
        aa.construct_grid_set()
        end = time.time()

        if kTimeCount:
            print("construct_grid_set: {}".format(end - start))

        start = time.time()
        shuffle(user_record)
        end = time.time()

        if kTimeCount:
            print("shuffle: {}".format(end - start))

        if args.algorithm_name == kPrivMDC:
            start = time.time()
            aa.define_user_mapping()
            end = time.time()
            if kTimeCount:
                print("define_user_mapping: {}".format(end - start))

        start = time.time()
        aa.run_ldp(ldp_records)
        end = time.time()
        if kTimeCount:
            print("run_ldp: {}".format(end - start))

        if args.kd and args.algorithm_name == kPrivMDC and args.split_ratio > 0:
            start = time.time()
            aa.extract_grids()
            end = time.time()
            if kTimeCount:
                print("extract_grids: {}".format(end - start))

        start = time.time()
        aa.get_consistent_grid_set()
        end = time.time()

        if kTimeCount:
            print("get_consistent_grid_set: {}".format(end - start))

        if args.algorithm_name == kFELIP or \
                args.algorithm_name == kFELIP_default or \
                args.algorithm_name == kFELIP_default_not_consistent or \
                args.algorithm_name == kPrivMDC or \
                args.algorithm_name == kPrivNud:
            start = time.time()
            aa.get_wu_for_2_way_group()
            end = time.time()
            if kTimeCount:
                print("get_wu_for_2_way_group: {}".format(end - start))

        if args.algorithm_name == kB_HDG:
            aa.get_weight_update_for_2_way_group()

        start = time.time()
        aa.answer_query_list(query_list.query_list)
        all_end = time.time()

        if kTimeCount:
            print("answer_query_list: {}".format(all_end - start))

        um = UM.UtilityMetric(args=args)
        mae_error = um.MAE(real_list=query_list.real_answer_list, est_list=aa.weighted_update_answer_list)

        # it_data = {"mae": mae_error, "eps": eps, "name": args.algorithm_name, "time": all_end - allstart}

        it_data = [mae_error, eps, args.algorithm_name, all_end - allstart]
        exp_log_list.append(it_data)

        error_repeat[r] = mae_error

        # if os.path.exists(ds_name):
        #     os.remove(ds_name)
        print("run took {}".format(all_end - allstart))
        print("progress {} {} : {}%".format(args.algorithm_name, eps, int((r+1)/repeat * 100)))

    mae_list.append(np.mean(error_repeat))
    std_list.append(np.std(error_repeat))
    percentile25_list.append(np.percentile(error_repeat, 25))
    percentile75_list.append(np.percentile(error_repeat, 75))

    print("algo: {} eps: {} done!".format(algo, eps))
    res = ExpResult(mae_list, std_list, percentile25_list, percentile75_list, exp_dataframe=exp_log_list)
    return res


def read_dataset(args):

    # list of lists
    user_record, domains_list, attr_types_list = load_dataset(txt_dataset_path=TXT_DATASET_PATH)

    random_seed = 1
    random.seed(random_seed)
    np.random.seed(seed=random_seed)

    query_list = GenQuery.QueryList(
        query_dimension=args.query_dimension,
        attr_num=args.attr_num,
        query_num=args.query_num,
        dimension_query_volume_list=kSelectivities,
        domains_list=domains_list,
        attr_types_list=attr_types_list,
        args=args)

    ok = True
    query_list.generate_query_list()
    while ok:
        ok = query_list.generate_real_answer_list(user_record)
        if ok:
            args.query_num -= 1
            query_list.query_num = args.query_num
            if args.query_num == 0:
                raise Exception("ZERO NUMBER OF QUERIES")

    code = "cat1ord1"
    if args.query_dimension == 4:
        code = "cat2ord2"
    elif args.query_dimension == 3:
        code = "ord3"

    txt_file_path = "e_exp_output/queries/query_" + DATASET + "_" + code + "_" + DATA_TYPE + ".txt"
    with open(txt_file_path, "w") as txt_fr_out:
        query_list.print_query_list(txt_fr_out)

    txt_file_path = "e_exp_output/answers/answers_" + DATASET + "_lambda=" + str(args.query_dimension) + "_type=" + DATA_TYPE + ".txt"
    with open(txt_file_path, "w") as txt_fr_out:
        query_list.print_query_answers(txt_fr_out)

    np.random.shuffle(user_record)
    return user_record, query_list, domains_list, attr_types_list


if __name__ == '__main__':

    main_log = []
    algo_l = []
    #eps_l = [0.1]
    #
    eps_l = [0.1, 0.5, 1.0, 1.5, 2.0]
    times_r = 20

    algo_l += [kPrivMDC] * len(eps_l)
    # algo_l += [kPrivNud] * len(eps_l)
    #algo_l += [kFELIP] * len(eps_l)
    # algo_l += [kB_HDG] * len(eps_l)
    # eps_l += eps_l
    # eps_l += eps_l
    # eps_l += eps_l

    plot_args = para.generate_args()  # define the parameters
    setup_args(args=plot_args)  # setup the parameters
    print("Repeat: {}\nVolume: {}".format(times_r, plot_args.dimension_query_volume))

    start = time.time()
    u_ser_record, query_list, domain_list, attr_types_list = read_dataset(plot_args)
    end = time.time()
    if kTimeCount:
        print("read dataset: {}".format(end - start))

    plot_data = {kFELIP_default: [], kFELIP: [],  kB_HDG: [], kB_TDG: [], kHBS: [], kPrivMDC: [], kPrivNud: []}
    config_list = {kFELIP_default: [], kFELIP: [],  kB_HDG: [], kB_TDG: [], kHBS: [], kPrivMDC: [], kPrivNud: []}
    total_mae_list = {kFELIP_default: [], kFELIP: [], kB_HDG: [], kB_TDG: [], kHBS: [], kPrivMDC: [], kPrivNud: []}

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        future_to_config = {executor.submit(sys_test, eps, algo, times_r,
                                            u_ser_record, query_list,
                                            domain_list, attr_types_list):
                                [algo, eps] for algo, eps in zip(algo_l, eps_l)}
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                data = future.result()
                maes = data.maes
                stds = data.stds
                percentile25_list = data.percentile25_list
                percentile75_list = data.percentile75_list
                #exp_dataframe = data.exp_dataframe
                with df_lock:
                    exp_dataframe = data.exp_dataframe
                    for item in exp_dataframe:
                        main_log.append(item)

            except Exception as exc:
                print('%r Problema future: %s' % (config, exc))
            else:
                plot_data[str(config[0])].append([maes, stds, percentile25_list,
                                                  percentile75_list, exp_dataframe,
                                                  str(config[0]) + " phi=" + str(config[1])])

                print("Config: {} Total mae: {:.12f}".format(config, np.sum(maes)))
                #print("len(log): {}".format(len(exp_dataframe)))
                total_mae_list[str(config[0])].append(np.sum(maes))
                config_list[str(config[0])].append(str(config[0]) + " " + str(config[1]))
                print("e: {}".format(eps_l))
                print("maes:")
                for e, mae, std in zip(eps_l, maes, stds):
                    print("{:.12f}".format(mae))

                print("var:")
                for e, mae, std in zip(eps_l, maes, stds):
                    print("{:.12f}".format(std))

                print("25p:")
                for p25 in percentile25_list:
                    print("{:.12f}".format(p25))

                print("75p:")
                for p75 in percentile75_list:
                    print("{:.12f}".format(p75))

    main_df = pd.DataFrame(main_log, columns=["mae", "eps", "name", "time"])
    plot.plot_std(sns_dataset=main_df, name="plot_eps_exp_ds{}_bn{}_sr{}_l{}_r{}".format(DATASET,
                                                                                         plot_args.bn_degree,
                                                                             plot_args.split_ratio,
                                                                             plot_args.query_dimension, times_r))
    main_df.to_csv("e_exp_output/log_csv/log_" + DATASET + "_" + DATA_TYPE
                   + "_" + str(plot_args.dimension_query_volume) + "_" + "dps=" + str(plot_args.split_ratio)
                   + ".csv", index=False)
