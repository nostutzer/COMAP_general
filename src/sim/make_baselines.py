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


t = time.time()
destr = Destriper()

if destr.perform_split:
    #freq_idx = range(18 * 4 * 64)
    freq_idx = [38]
else:
    freq_idx = range(4 * 64)


t0 = time.time()

if not destr.perform_split:
    print("Loading data and initializing pointing:")
    destr.get_data()
    destr.initialize_P_and_F()
print("Loading time:", time.time() - t0, "sec")

t0 = time.time()
print("Looping over frequencies:")

def dummy(idx):
    print("\n", "Processing frequency number:", idx, "\n")
    t = time.time()
    
    if destr.perform_split:
        feed, sb, freq = destr.all_idx[:, freq_idx]

        print("hei", feed, sb, freq)
        
        current_batch_def = destr.batch_def[:, feed, sb][:, 0]
        destr.unique_batches, destr.indices = np.unique(destr.batch_def[:, feed, sb], return_inverse = True)
        destr.N_batch_per_freq = destr.unique_batches.shape[0]
        
        destr.batch_buffer = [[] for i in range(destr.N_batch_per_freq)]

        for i in range(1, destr.N_batch_per_freq):
            #print(destr.split_scans[current_batch_def == destr.unique_batches[i]])
            destr.currentNames = destr.names[current_batch_def == destr.unique_batches[i]]
            
            destr.run(feed, sb, freq)
            
            print("Destriping batch:")
            destr.make_baseline_only()
            print(destr.baseline_tod)
        sys.exit()
    
    else:        
        #if baseline_tod.shape[0] * 4 * 64 >= 2147483647:
        #print((destr.sb, destr.freq, baseline_tod))
        #destr.save_baseline_tod_per_freq(destr.sb, destr.freq, destr.baseline_tod)
        initem = [destr.sb, destr.freq, destr.baseline_tod]
        dummy.q.put(initem)
        dummy.iterr.value += 1
        print("Frequency loop progress: ", dummy.iterr.value / 256 * 100, "%")
        return None
        #return [destr.sb, destr.freq, baseline_tod]
        #else:
        #    return baseline_tod

def dummy_init(q, lock, iterr):
    dummy.q = q
    dummy.lock = lock
    dummy.iterr = iterr

def dummy_save():
    with dummy.lock:
        #print("Inside saver")
        
        while dummy.iterr.value <= 256 or not dummy.q.empty():
            outitem = dummy.q.get()
            #destr.save_baseline_tod_per_freq(outitem[0], outitem[1], outitem[2])
            print("Saving baselines for sb and freq number:", outitem[0], outitem[1])

m = multiproc.Manager()
q = m.Queue()
lock = m.Lock()
iterr = multiproc.Value("i", 0)

with multiproc.Pool(destr.Nproc, dummy_init, [q, lock, iterr]) as pool:
    #pool.apply_async(dummy_save)
    baselines = pool.map(dummy, freq_idx)

pool.close()
pool.join()
print("Finished frequency loop:", time.time() - t0, "sec")

#baselines = np.array(baselines)
#baselines = baselines[:, 0, :]
#baselines = baselines.transpose()

#destr.save_baseline_tod(baselines)

print("Time: ", time.time() - t0, "sec")
