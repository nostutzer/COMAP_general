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
from tqdm import trange
import ctypes 

t = time.time()
destr = Destriper()

if destr.perform_split:
    freq_idx = range(19 * 4 * 64)
    #freq_idx = [1, 10]
    #freq_idx = [0]
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
            #print("hei")
            n0 = 20
            n1 = 4
            n2 = 64
            n3 = destr.N_buffer.shape[0]

            prod1 = n2 * n3
            prod2 = prod1 * n1
            for j in range(len(destr.currentNames)):
                #tr = time.time()
                alive = destr.feeds_alive[destr.currentNames[j]]
                N_alive = alive.shape[0]
                #print("hei1", j, len(destr.currentNames), time.time() - tr, "sec")
                #tr = time.time()
                feed_idx = np.where(alive == feed)[0]
                #print("hei2", j, len(destr.currentNames), time.time() - tr, "sec")
                #tr = time.time()
                if feed_idx.size == 1:
                    #destr.a_buffer[feed_idx, sb, freq, destr.name_buffer == destr.currentNames[j]] = destr.a[destr.scan_per_baseline == destr.currentNames[j]]
                    idx1 = np.where(destr.name_buffer == destr.currentNames[j])[0]
                    idx2 = np.where(destr.scan_per_baseline == destr.currentNames[j])[0]

                    for k in range(len(idx1)):
                        idx = int(prod2 * feed_idx + prod1 * sb + n3 * freq + idx1[k])
                        dummy.buffer[idx] = destr.a[idx2[k]]
                    #print("hei3", j, len(destr.currentNames), time.time() - tr, "sec")
                    #tr = time.time()

                #print("hei4", j, len(destr.currentNames), time.time() - tr, "sec")
                if destr.currentNames[j] == "co6_001197410.h5":
                    with dummy.lock:
                        tod_lens = destr.tod_lens
                        
                        outfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/wo_sim/highpass/002Hz/default/large_dataset/masked/baselines/allsamples/"
                
                        #print("Saveing baselines to:", outfile_path)
                        if not os.path.exists(outfile_path):
                            os.mkdir(outfile_path)

                        baseline_buffer = destr.F.dot(destr.a)
                        start, stop = destr.start_stop[:, j]
                        baseline      = baseline_buffer[start:stop]
                        #print(start, stop)
                        
                        
                        baseline = baseline.astype(np.float32)
                        
                        new_name = destr.currentNames[j].split(".")
                        new_name = new_name[0] + "_temp." + new_name[1]
                        if baseline.size > 0:
                            #print(new_name, destr.currentNames[j], "co6_001197410.h5", baseline.shape)
                            outfile = h5py.File(outfile_path + new_name, "a")
                            if "tod_baseline" not in outfile.keys():
                                outfile.create_dataset("tod_baseline", data = np.zeros((N_alive, 4, 64, baseline.shape[0]), dtype = np.float32), dtype = "float32")
                            
                            data = outfile["tod_baseline"]
                            
                            data[feed_idx[0], sb, freq, :] = baseline

                            outfile.close()
            #with dummy.lock:
            #destr.save_baseline_tod_per_batch(dummy.lock)
            #print("Save batch time:", time.time() - td, "sec")
        
            #print("Batch loop time:", time.time() - ti, "sec", "Before Saver time:", t_before, "sec", "N batches:", destr.N_batch_per_freq, "N In Batch:", len(destr.currentNames))
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

def dummy_init(q, lock, iterr, buffer):
    dummy.q = q
    dummy.lock = lock
    dummy.iterr = iterr
    dummy.buffer = buffer

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
buffer = multiproc.RawArray(ctypes.c_float, (20 * 4 * 64 * destr.N_buffer.shape[0]))

with multiproc.Pool(destr.Nproc, dummy_init, [q, lock, iterr, buffer]) as pool:
    baselines = pool.map(dummy, freq_idx)
    #pool.apply_async(dummy_save)

pool.close()
pool.join()

destr.a_buffer = np.frombuffer(buffer, dtype = np.float32).reshape(20, 4, 64, destr.N_buffer.shape[0])

print("Finished frequency loop:", time.time() - t0, "sec")
#destr.save_from_queue(q)

destr.save_baselines_from_buffer()

#baselines = np.array(baselines)
#baselines = baselines[:, 0, :]
#baselines = baselines.transpose()

#destr.save_baseline_tod(baselines)

print("Time: ", time.time() - t0, "sec")
