"""
Helper functions
"""

import simpy
import numpy as np
from des import source

def des(n, new_costumers=500, arrival=50, capacity=51):
    """
    Run DES model
    """
    env = simpy.Environment()
    wait_times = np.zeros(new_costumers)

    # Set up simpy simulation
    env = simpy.Environment()
    arrival_rate = (arrival)/n  # Generate new customers roughly every x seconds
    counter = simpy.Resource(env, capacity=n)
    env.process(source(env, new_costumers, arrival_rate, counter, capacity))
    env.run()
    load = n * (arrival_rate / capacity)
    print(f"Amount of counters: {n}")
    print(f"Load on machines: {load}")
    print(f'Average wait time per person: {np.sum(wait_times)/new_costumers}')
