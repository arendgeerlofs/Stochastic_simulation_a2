"""
Helper functions
"""
import random
import re
import simpy
import numpy as np
import matplotlib.pyplot as plt

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
    tib = 0
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
    service_times[int(number)] = tib
    sojurn_rates[int(number)] = (wait+tib)/tib


def des(n, new_customers, rho, mu, fifo, st_distribution):
    """
    Run DES model
    """
    env = simpy.Environment()

    # Check input distribution
    if st_distribution not in ['M', 'D', 'LongTail']:
        print("Service time distribution unknown, try again!")
        return

    # Find lambda
    arrival_rate = rho*mu*n  # Generate new customers roughly every 1/arrival rate seconds

    # Set up data_arrays
    global wait_times
    global service_times
    global sojurn_rates
    wait_times = np.zeros(new_customers)
    service_times = np.zeros(new_customers)
    sojurn_rates = np.zeros(new_customers)

    # Set up scheduling type
    if fifo:
        counter = simpy.Resource(env, n)
    else:
        counter = simpy.PriorityResource(env, n)

    # Run DES process
    env.process(source(env, new_customers, arrival_rate, counter, mu, fifo, st_distribution))
    env.run()
    #print(f"Amount of counters: {n}")
    #print(f"Load on machines: {rho}")
    #print(f'Average wait time per person: {np.sum(wait_times)/new_costumers}')
    #plt.hist(wait_times)
    #plt.show()
    return wait_times, service_times, sojurn_rates

def steady_state_plot(customers, runs, rho_values, mu=1, fifo=True, st_distribution='M'):
    """
    Plot average wait time for different amounts of customers and runs
    Used to show convergence to steady state
    """
    for rho_index, rho in enumerate(rho_values):
        data = np.zeros((3, int(customers), runs))
        for index_j, j in enumerate([1, 2, 4]):
            for c in range(0, customers):
                for i in range(runs):
                    data[index_j][int(c)][i] = np.mean(des(j, c+1, rho, mu, fifo, st_distribution)[0])
        means = np.mean(data, axis=2)
        stds = np.std(data, axis=2)
        xdata = np.linspace(1, customers, int(customers))
        plt.plot(xdata, means[0], 'r', label='n=1')
        plt.plot(xdata, means[1], 'b', label='n=2')
        plt.plot(xdata, means[2], 'g', label='n=4')
        plt.fill_between(xdata, means[0]-stds[0], means[0]+stds[0], color='r', alpha=.5)
        plt.fill_between(xdata, means[1]-stds[1], means[1]+stds[1], color='b', alpha=.5)
        plt.fill_between(xdata, means[2]-stds[2], means[2]+stds[2], color='g', alpha=.5)
        plt.legend()
        plt.xlabel("Amount of customers")
        plt.ylabel("Average waiting time per customer")
        plt.savefig(f"test{rho_index}", dpi=300)
        plt.show()

def histograms(customers, runs, rho_values, queues=1, mu=1, fifo=True, st_distribution='M', name="hist"):
    for rho in rho_values:
        wait_times_run = np.zeros((runs, customers))
        service_times_run = np.zeros((runs, customers))
        sojurn_rates_run = np.zeros((runs, customers))
        for i in range(runs):
            wait_time, service_time, sojurn_rate = des(queues, customers, rho, mu, fifo, st_distribution)
            wait_times_run[i] = wait_time
            service_times_run[i] = service_time
            sojurn_rates_run[i] = sojurn_rate
        wait_times_run = np.mean(wait_times_run, axis=0)
        service_times_run = np.mean(service_times_run, axis=0)
        sojurn_rates_run = np.mean(sojurn_rates_run, axis=0)
        plt.hist(wait_times_run, 20)
        plt.xlabel("Wait time bins")
        plt.ylabel("Amount of customers")
        plt.savefig(f"wait_{name}", dpi=300)
        plt.show()
        plt.hist(service_times_run, 20)
        plt.xlabel("Service time bins")
        plt.ylabel("Amount of customers")
        plt.savefig(f"service_{name}", dpi=300)
        plt.show()
        plt.hist(sojurn_rates_run, 20)
        plt.xlabel("Sojurn time / service time bins")
        plt.ylabel("Amount of customers")
        plt.savefig(f"sojurn_{name}", dpi=300)
        plt.show()
