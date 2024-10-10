import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# FOLDER_NAME="results"
#
# def plot_lines(eps, plot_data, args):
#     markers = ['o','v']
#     for pd in plot_data:
#         e = eps
#         b = pd[0]
#         plt.plot(eps, pd[0], marker='o', label=pd[1])
#
#     plt.xlabel('Epsilon')
#     plt.ylabel("MAE")
#     plt.title("N={} d={} lambda={} q_num={}".format(args.user_num,
#                                                     args.domain_size,
#                                                     args.query_dimension,
#                                                     args.query_num))
#     plt.yscale("log")
#     plt.legend()
#     plt.show()
#
# def plot_bar(configs, total_maes):
#     fig = plt.figure()
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.bar(configs, total_maes)
#     plt.xlabel('Config')
#     plt.ylabel("Total MAE")
#     plt.yscale("log")
#     plt.legend()
#     plt.show()

def plot_std(sns_dataset, name):
    r = sns.lineplot(x='eps', y='mae', data=sns_dataset, hue='name', markers=True,
                     dashes=False, style="name", err_style="bars", errorbar=("se", 2))
    r.set(yscale='log')
    plt.show()
    plt.savefig("e_exp_output/plots_fig/{}.png".format(name))

if __name__ == '__main__':
    DATASET = "bfive"
    DATA_TYPE = "num"
    exp_dataframe = pd.read_csv("e_exp_output/log_csv/log_" + DATASET + "_" + DATA_TYPE + "_0.7_" + ".csv")
    plot_std(exp_dataframe)
