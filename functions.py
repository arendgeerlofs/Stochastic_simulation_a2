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

