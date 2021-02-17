import numpy as np 
from scipy.sparse import csc_matrix, identity, diags
import sparse_dot_mkl
import scipy.sparse.linalg as linalg
from scipy import signal
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as multiproc
import time
import sys
import h5py
import WCS
import copy
import os
import re
import argparse
from destr import Destriper 

freq_idx = range(4 * 64)

t = time.time()
destr = Destriper()
print("Loading data and initializing pointing:")
t0 = time.time()
destr.get_data()
destr.initialize_P_and_F()
print("Loading time:", time.time() - t0, "sec")

t0 = time.time()
print("Looping over frequencies:")

def dummy(idx):
    print("\n", "Processing frequency number:", idx, "\n")
    t = time.time()
    destr.run(freq_idx = idx)

    destr.make_baseline_only()
    return np.array([destr.baseline_tod])
    
with multiproc.Pool(processes = destr.Nproc) as pool:
    baselines = pool.map(dummy, freq_idx)
pool.close()
pool.join()
print("Finished frequency loop:", time.time() - t0, "sec")

baselines = np.array(baselines)
baselines = baselines[:, 0, :]
baselines = baselines.transpose()

#destr.save_baseline_tod(baselines)
print("Time: ", time.time() - t0, "sec")
    
    

