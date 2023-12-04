"""
Where to run simulations from
"""
import numpy as np
import scipy.stats as stats
from functions import des, steady_state_plot, histograms, stats_ns_waittimes, run_fifo_sjf, plot_fifo_sjf
CUSTOMERS = 2500
RUNS = 100
RHO_VALUES = [0.99, 0.9, 0.75, 0.5]
steady_state_plot(CUSTOMERS, RUNS, RHO_VALUES, mu=0.5, st_distribution='LongTail')
histograms(CUSTOMERS, 1, [0.9], queues=1, st_dist='M', name="MMN")
histograms(CUSTOMERS, 1, [0.9], queues=1, st_dist='D', name="MDN")
histograms(CUSTOMERS, 1, [0.9], mu=0.5, queues=1, st_dist='LongTail', name="LT")

wait_times = np.zeros(RUNS)
for i in range(RUNS):
    wait_times[i] = np.mean(des(1, CUSTOMERS, 0.5, 1, True, 'M')[0])
print(stats.ttest_1samp(wait_times, 1))

t_table = np.zeros((12, 12))
wait_data = np.zeros((12, RUNS))
for i in range(RUNS):
    wait_data[0][i] = np.mean(des(1, CUSTOMERS, 0.9, 1, True, 'M')[0])
    wait_data[1][i] = np.mean(des(2, CUSTOMERS, 0.9, 1, True, 'M')[0])
    wait_data[2][i] = np.mean(des(4, CUSTOMERS, 0.9, 1, True, 'M')[0])
    wait_data[3][i] = np.mean(des(1, CUSTOMERS, 0.9, 1, False, 'M')[0])
    wait_data[4][i] = np.mean(des(2, CUSTOMERS, 0.9, 1, False, 'M')[0])
    wait_data[5][i] = np.mean(des(4, CUSTOMERS, 0.9, 1, False, 'M')[0])
    wait_data[6][i] = np.mean(des(1, CUSTOMERS, 0.9, 1, True, 'D')[0])
    wait_data[7][i] = np.mean(des(2, CUSTOMERS, 0.9, 1, True, 'D')[0])
    wait_data[8][i] = np.mean(des(4, CUSTOMERS, 0.9, 1, True, 'D')[0])
    wait_data[9][i] = np.mean(des(1, CUSTOMERS, 0.9, 0.5, True, 'LongTail')[0])
    wait_data[10][i] = np.mean(des(2, CUSTOMERS, 0.9, 0.5, True, 'LongTail')[0])
    wait_data[11][i] = np.mean(des(4, CUSTOMERS, 0.9, 0.5, True, 'LongTail')[0])
for i in range(12):
    for j in range(12):
        if i != j:
            t_table[i][j] =stats.ttest_ind(wait_data[i], wait_data[j], equal_var=False)[0]
print(t_table)

n_values = [1,2,4]
stats_ns_waittimes(CUSTOMERS, RUNS, RHO_VALUES, n_values, mu=1, fifo=True, st_distribution='M')
data_mean, data_CI = run_fifo_sjf(CUSTOMERS, RUNS, RHO_VALUES, n = 1, mu=1, st_distribution='M')
plot_fifo_sjf(RHO_VALUES, data_mean, data_CI)