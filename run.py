"""
Where to run simulations from
"""
import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats
from functions import des, steady_state_plot
CUSTOMERS = 500
RUNS = 50
RHO_VALUES = [0.25, 0.5, 0.75, 0.99]
steady_state_plot(CUSTOMERS, RUNS, RHO_VALUES)
