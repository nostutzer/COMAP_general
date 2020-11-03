import numpy as np
import h5py
import matplotlib.pyplot as plt
import WCS
import time
import shutil
from tqdm import trange
import sys

t0 = time.time()

data_path1 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level1/2020-07/"
data_name1 = data_path1 + "comap-0015330-2020-07-31-040632.hd5"

data_path2 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level1/2020-07/sim/"
data_name2 = data_path2 + "comap-0015330-2020-07-31-040632.hd5"

print("Load infiles: ", time.time() - t0, " sec")
infile1  = h5py.File(data_name1, "r")
tod1     = np.array(infile1["/spectrometer/tod"]) 
MJD1     = np.array(infile1["/spectrometer/MJD"]) 


infile2  = h5py.File(data_name2, "r")
tod2     = np.array(infile2["/spectrometer/tod"]) 
MJD2     = np.array(infile2["/spectrometer/MJD"]) 

print("Infiles loaded: ", time.time() - t0, " sec")

fig, ax = plt.subplots(4, 3, sharex = "col", sharey = "row")
for i in range(18):
    for j in range(4):
        ax[j, 0].plot(tod1[i, j, 125, :])
        
        ax[j, 1].plot(tod2[i, j, 125, :])
        
        ax[j, 2].plot(tod3[i, j, 125, :])

ax[0, 0].set_title("obsID: 15330 without sim")
ax[0, 1].set_title("obsID: 15330 with sim")
ax[0, 2].set_title("obsID: 15330 with sim")

ax[-1, 0].set_xlabel("MJD")
ax[-1, 1].set_xlabel("MJD")
ax[-1, 2].set_xlabel("MJD")

ax[0, 0].set_ylabel("TOD")
ax[0, 1].set_ylabel("TOD")
ax[0, 2].set_ylabel("TOD")

plt.savefig("tod.png")

