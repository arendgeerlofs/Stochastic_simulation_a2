"""
Where to run simulations from
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
#from scipy import stats
from functions import des, steady_state_plot, histograms
CUSTOMERS = 1000
RUNS = 100
RHO_VALUES = [0.99, 0.9, 0.75, 0.5]
# steady_state_plot(CUSTOMERS, RUNS, RHO_VALUES, mu=10, st_distribution='LongTail')
# histograms(CUSTOMERS, RUNS, [0.9], queues=1, st_distribution='M', name="MMN")
# histograms(CUSTOMERS, RUNS, [0.9], queues=1, st_distribution='D', name="MDN")
histograms(CUSTOMERS, RUNS, [0.9], mu=5, queues=1, st_distribution='LongTail', name="LT")

# wait_times = np.zeros(RUNS)
# for i in range(RUNS):
#     wait_times[i] = np.mean(des(1, CUSTOMERS, 0.5, 1, True, 'M')[0])
# print(stats.ttest_1samp(wait_times, 1))
