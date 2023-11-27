"""
Helper functions
"""

import simpy
import numpy as np
import random
import re
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
    print(f"Amount of counters: {n}")
    print(f"Load on machines: {rho}")
    print(f'Average wait time per person: {np.sum(wait_times)/new_costumers}')
    plt.hist(wait_times)
    plt.show()