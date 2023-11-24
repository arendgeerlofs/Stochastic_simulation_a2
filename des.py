"""
DES functions
"""

import random
import re

def source(env, number, interval, counter, time_needed):
    """Source generates customers randomly"""
    for i in range(number):
        c = customer(env, f'Customer{i:2d}', counter, time_in_bank=time_needed)
        env.process(c)
        t = random.expovariate(1 / interval)
        yield env.timeout(t)


def customer(env, name, counter, time_in_bank):
    """Customer arrives, is served and leaves."""
    arrive = env.now
    with counter.request() as req:
        # Wait for the counter
        yield req
        print(f'finish time:{env.now}')
        print(f'arrive : {arrive}')
        wait = env.now - arrive

        # We got to the counter

        tib = random.expovariate(1.0 / time_in_bank)
        yield env.timeout(tib)
    number = re.findall("d+", name)[0]
    print(f'wait time: {wait}')
    wait_times[int(number)] = wait
