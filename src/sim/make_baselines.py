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
    freq_idx = range(19 * 4 * 64)
    #freq_idx = [1, 10]
    #freq_idx = [19 * 4 * 64 - 3, 19 * 4 * 64 - 2, 19 * 4 * 64 - 1]
else:
    freq_idx = range(4 * 64)
    #freq_idx = [0]

t0 = time.time()

if not destr.perform_split:
    print("Loading data and initializing pointing:")
    destr.get_data()
    destr.initialize_P_and_F()
    print("Loading time:", time.time() - t0, "sec")

t0 = time.time()
print("Looping over frequencies:")

def dummy(idx):
    #print("\n", "Processing frequency number:", idx, "\n")
    t = time.time()

    if destr.perform_split:
        feed, sb, freq = destr.all_idx[:, idx]
        feed, sb, freq = int(feed), int(sb), int(freq)
        destr.freq_idx = idx
        current_batch_def = destr.batch_def[:, feed, sb]
        destr.unique_batches, destr.indices = np.unique(destr.batch_def[:, feed, sb], return_inverse = True)
        destr.N_batch_per_freq = destr.unique_batches.shape[0]
        
        destr.batch_buffer = [[] for i in range(destr.N_batch_per_freq)]

        for i in range(destr.N_batch_per_freq):
            #print(destr.split_scans[current_batch_def == destr.unique_batches[i]])
            ti = time.time()
        
            destr.currentNames = destr.names[current_batch_def == destr.unique_batches[i]]
            #print("Get batch names:", time.time() - td, "sec")
            #td = time.time()
            destr.run(feed, sb, freq)
            #print("Run func time: ", time.time() - td, "sec")

            #td = time.time()
            
            destr.make_baseline_only()
            #print("Destriper time:", time.time() - td, "sec")
            #names  = destr.currentNames
            #start_stop = destr.start_stop
            #initem = [feed, sb, freq, destr.currentNames, destr.start_stop, destr.a, destr.Nperbaselines, destr.scan_per_baseline]
            
            #dummy.q.put(initem)
            t_before = time.time() - ti
            #td = time.time()
            #with dummy.lock:
            destr.save_baseline_tod_per_batch(dummy.lock)
            #print("Save batch time:", time.time() - td, "sec")
        
            print("Batch loop time:", time.time() - ti, "sec", "Before Saver time:", t_before, "sec", "N batches:", destr.N_batch_per_freq, "N In Batch:", len(destr.currentNames))
            """if not dummy.q.empty():
                with dummy.lock:
                    outitem = dummy.q.get()

                    #print("Saving baselines for feed, sb and freq number:", outitem[0], outitem[1], outitem[2])
                    
                    destr.save_baseline_tod_per_batch(outitem[0], outitem[1], outitem[2], outitem[3])"""
    else:        
        destr.run(freq_idx = idx)

        destr.make_baseline_only()
        

        initem = [destr.sb, destr.freq, destr.baseline_tod]
        dummy.q.put(initem)        

        if not dummy.q.empty():
            with dummy.lock:
                outitem = dummy.q.get()
                print("Saving baselines for sb and freq number:", outitem[0], outitem[1])
                destr.save_baseline_tod_per_freq(outitem[0], outitem[1], outitem[2])

    with dummy.lock:
        print("Frequency loop progress: ",  dummy.iterr.value, "/", len(freq_idx), "|", round(dummy.iterr.value / len(freq_idx) * 100, 4), "%")
        dummy.iterr.value += 1

    return None

def dummy_init(q, lock, iterr):
    dummy.q = q
    dummy.lock = lock
    dummy.iterr = iterr

def dummy_save():
    print("Initializing saver:")
    destr.save_from_queue(dummy.q, iter)
    print("Finalize saver:")
    
    """
    with dummy.lock:
        #print("Inside saver")
        
        while dummy.iterr.value <= 256 or not dummy.q.empty():
            outitem = dummy.q.get()
            #destr.save_baseline_tod_per_freq(outitem[0], outitem[1], outitem[2])
            print("Saving baselines for sb and freq number:", outitem[0], outitem[1])
    """
m = multiproc.Manager()
q = m.Queue()
lock = m.Lock()
iterr = multiproc.Value("i", 0)

with multiproc.Pool(destr.Nproc, dummy_init, [q, lock, iterr]) as pool:
    baselines = pool.map(dummy, freq_idx)
    #pool.apply_async(dummy_save)

pool.close()
pool.join()

print("Finished frequency loop:", time.time() - t0, "sec")
#destr.save_from_queue(q)


#baselines = np.array(baselines)
#baselines = baselines[:, 0, :]
#baselines = baselines.transpose()

#destr.save_baseline_tod(baselines)

print("Time: ", time.time() - t0, "sec")
