"""
Helper functions
"""
import random
import re
import simpy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs
import math

def source(env, number, arrival_rate, counter, mu, fifo, st_distribution):
    """Source generates customers randomly"""
    for i in range(number):
        c = customer(env, f'Customer{i:2d}', counter, mu, fifo, st_distribution)
        env.process(c)
        t = random.expovariate(arrival_rate)
        yield env.timeout(t)


def customer(env, name, counter, mu, fifo, st_distribution):
    """Customer arrives, is served and leaves."""
    arrive = env.now

    # service time distributions
    if st_distribution == 'M':
        tib = random.expovariate(mu)
    elif st_distribution == 'D':
        tib = 1/mu
    elif st_distribution == 'LongTail':
        u = random.uniform(0,1)
        if u < 0.25:
            tib = random.expovariate(1/5)
        else:
            tib = random.expovariate(1)

    # Scheduling type
    if fifo:
        cr = counter.request()
    else:
        cr = counter.request(priority = tib)

    with cr as req:
        # Wait for the counter
        yield req
        start_time = env.now
        wait = start_time - arrive

        # We got to the counter
        yield env.timeout(tib)
    number = re.findall("\d+", name)[0]
    wait_times[int(number)] = wait


def des(n, new_costumers, rho, mu, fifo, st_distribution):
    """
    Run DES model
    """
    env = simpy.Environment()

    # Check input distribution
    if st_distribution not in ['M', 'D', 'LongTail']:
        print("Service time distribution unknown, try again!")
        return

    # Find lambda
    arrival_rate = rho*mu*n  # Generate new customers roughly every x seconds

    # Set up data_arrays
    global wait_times
    wait_times = np.zeros(new_costumers)

    # Set up scheduling type
    if fifo:
        counter = simpy.Resource(env, n)
    else:
        counter = simpy.PriorityResource(env, n)

    # Run DES process
    env.process(source(env, new_costumers, arrival_rate, counter, mu, fifo, st_distribution))
    env.run()
    #print(f"Amount of counters: {n}")
    #print(f"Load on machines: {rho}")
    #print(f'Average wait time per person: {np.sum(wait_times)/new_costumers}')
    #plt.hist(wait_times)
    #plt.show()
    return np.mean(wait_times)

def steady_state_plot(customers, runs, rho_values, mu=1, fifo=True, st_distribution='M'):
    """
    Plot average wait time for different amounts of customers and runs
    Used to show convergence to steady state
    """
    for rho_index, rho in enumerate(rho_values):
        data = np.zeros((3, customers, runs))
        for index_j, j in enumerate([1, 2, 4]):
            for c in range(customers):
                for i in range(runs):
                    data[index_j][c][i] = des(j, c+1, rho, mu, fifo, st_distribution)
        means = np.mean(data, axis=2)
        stds = np.std(data, axis=2)
        xdata = np.linspace(1, customers, customers)
        plt.plot(xdata, means[0], 'r', label='n=1')
        plt.plot(xdata, means[1], 'b', label='n=2')
        plt.plot(xdata, means[2], 'g', label='n=4')
        plt.fill_between(xdata, means[0]-stds[0], means[0]+stds[0], color='r', alpha=.5)
        plt.fill_between(xdata, means[1]-stds[1], means[1]+stds[1], color='b', alpha=.5)
        plt.fill_between(xdata, means[2]-stds[2], means[2]+stds[2], color='g', alpha=.5)
        plt.legend()
        plt.xlabel("Amount of customers")
        plt.ylabel("Average waiting time per customer")
        plt.title("Convergence of waiting time into steady state")
        plt.savefig(f"SS_{rho_index}", dpi=300)
        plt.show()


def stats_ns_waittimes(new_costumers, runs, rhos, n_values, mu=1, fifo=True, st_distribution='M'):
    '''
    Excectues DES for every value of rho, n and for a number of runs. 
    This function also does appropriate hypothesis testing for every value of rho.
    '''
    data = np.zeros((len(rhos), runs, len(n_values)))
    data_mean = np.zeros((len(rhos), len(n_values)))
    data_CI = np.zeros((len(rhos), len(n_values), 2))

    for rho_index, rho in enumerate(rhos):
        for i in range(runs):
            for n_index, n in enumerate(n_values):
                # Running the model
                des(n, new_costumers, rho, mu, fifo, st_distribution)
                data[rho_index, i, n_index] = np.mean(wait_times[1000:])
            data_mean[rho_index, n_index] = np.mean(data[rho_index, :i, n_index], axis=0)
            if i > 1:
                data_CI[rho_index, n_index, :] = np.transpose(np.percentile(data[rho_index, :i, n_index], [2.5, 97.5], axis=0))

        # Check whether we have to do an One-way ANOVA or a Kruskal-Wallis H-test
        _, p_normal = stats.normaltest(data[rho_index], axis=0)
        _, p_levene = stats.levene(data[rho_index, :, 0], data[rho_index, :, 1], data[rho_index, :, 2])
        print(f'for rho = {rho}, the p_normal values are for n=1, n=2 and n=4 respectively: {p_normal}')
        print(f'For rho = {rho}, the p_levene = {p_levene}')
        # Report One-way ANOVA if all the values of p_normal and p_levene are greater than 0.05. 
        # If not, report Kruskal-Wallis H-test
        F, p_anova = stats.f_oneway(data[rho_index][:,0], data[rho_index][:,1], data[rho_index][:,2])
        print(f'One-way Anova test for rho = {rho} with F = {F} and p = {p_anova}')
        # For no value we came accross, all conditions for one-way ANOVA were met.
        # If they were, a post-hoc test corresponding to a significant ANOVA test would also have been performed and reported.  
        H, p_kruskal = stats.kruskal(data[rho_index][:,0], data[rho_index][:,1], data[rho_index][:,2])
        print(f'Kruskal-Wallis H-test for rho = {rho} with H = {H} and p = {p_kruskal}')
        p_values_dunn = scikit_posthocs.posthoc_dunn([data[rho_index][:,0], data[rho_index][:,1], data[rho_index][:,2]], p_adjust = 'bonferroni')
        print(f'The p-values of the post_hoc Dunn\'s test are: {p_values_dunn}')


def run_fifo_sjf(new_costumers, runs, rhos, n = 1, mu=1, st_distribution='M'):
    '''
    Run DES for FIFO and SJF for every value of rho and for a number of runs.
    This function also does the appropriate hypothesis testing to determine if FIFO and SJF significantly differ.
    It does hypothesis testing for every given value of rho.
    Remark: n cannot contain multiple values.
    '''

    # For sjf and fifo run the model and statistically compare wait_times. 
    fifos = [True, False]
    data = np.zeros((len(fifos), len(rhos), runs))
    data_mean = np.zeros((len(fifos), len(rhos)))
    data_CI = np.zeros((len(fifos), len(rhos), 2))
    p_normal = np.zeros((len(rhos), len(fifos)))
    p_levene = np.zeros((len(rhos)))
    T = np.zeros((len(rhos)))
    p_ttest = np.zeros((len(rhos)))
    U = np.zeros((len(rhos)))
    p_MW = np.zeros((len(rhos)))

    for rho_index, rho in enumerate(rhos):
        for fifo_index, fifo in enumerate(fifos):
            for run in range(runs):
                des(n, new_costumers, rho, mu, fifo, st_distribution)
                data[fifo_index, rho_index, run] = np.mean(wait_times[1000:])
            data_mean[fifo_index, rho_index] = np.mean(data[fifo_index, rho_index, :])
            data_CI[fifo_index, rho_index, :] = np.percentile(data[fifo_index, rho_index, :], [2.5, 97.5])
            _, p_normal[rho_index, fifo_index] = stats.normaltest(data[fifo_index, rho_index], axis=0)
            print(f'for fifo = {fifo} en rho = {rho}, the p_normal value is = {p_normal[rho_index, fifo_index]}')
        _, p_levene[rho_index] = stats.levene(data[0, rho_index, :], data[1, rho_index, :])
        # If both values of p_normal and p_levene are above 0.05, The independent t-test can be reported.
        # If not, then report the Mann-Whitney U-value and and p-value
        print(f'For rho = {rho}, the Levene\'s test between fifo and sjf is p_levene = {p_levene[rho_index]}')
        T[rho_index], p_ttest[rho_index] = stats.ttest_ind(data[0, rho_index, :], data[1, rho_index, :])
        print(f'For rho = {rho}, the T-test has a value of {T[rho_index]} with a significance of p_ttest = {p_ttest[rho_index]}')
        U[rho_index], p_MW[rho_index] = stats.mannwhitneyu(data[0, rho_index, :], data[1, rho_index, :])
        print(f'For rho = {rho}, the Mann-Whitney U-test has a value of {U[rho_index]} with a significance of p_ttest = {p_MW[rho_index]}')R

def plot_fifo_sjf(rhos, data_mean, data_CI):
    '''
    Given the rho-values, mean data of DES over multiple runs, and 95% confidence interval over multiple runs
    This function plots the mean and confidence intervals agains the values of rho.
    Remark: It plots only for a single value of n and for fifo = [True, False]
    '''
    plt.figure()
    plt.plot(rhos, data_mean[0], label='FIFO', color='b', marker='x')
    plt.plot(rhos, data_mean[1], label='sjf', color = 'g', marker='x')
    plt.fill_between(rhos, data_CI[0, :, 0], data_CI[0, :, 1], color='b', alpha=0.5)
    plt.fill_between(rhos, data_CI[1, :, 0], data_CI[1, :, 1], color='g', alpha=0.5)
    plt.legend()
    plt.xlabel('Rho')
    plt.ylabel('Average waiting time')
    plt.savefig('fifo vs sjf for all rhos from rho is one eighth')
    plt.show()