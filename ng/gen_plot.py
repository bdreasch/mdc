import scipy.stats as stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

exp_dataframe = pd.read_csv("ipums.csv")
r = sns.lineplot(x='eps', y='mae', data=exp_dataframe, hue='name')
r.set(yscale='log')
plt.show()