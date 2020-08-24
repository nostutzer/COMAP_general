import numpy as np
import h5py
import matplotlib.pyplot as plt
import WCS
import time
import shutil
from tqdm import trange

cube_path = "/mn/stornext/d16/cmbco/comap/protodir/"
cube_filename = cube_path + "cube_real.npy"
cube = np.load(cube_filename)
cubeshape = cube.shape
cube = cube.reshape(cubeshape[0], cubeshape[0], 4, 1024)

data_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/"
data_name = data_path + "tod_sim.hd5"

infile  = h5py.File(data_name, "r")
tod     = np.array(infile["/spectrometer/tod"]) 

ra     = np.array(infile["/spectrometer/pixel_pointing/pixel_ra"]) 
dec     = np.array(infile["/spectrometer/pixel_pointing/pixel_dec"]) 

print(ra[0, :])
print(ra[10, :])
print(tod.shape)

"""
x = np.arange(0, len(tod[0, 0, 0, :]))
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(cube[:, :, 0, 0])
ax[0, 1].imshow(cube[:, :, 1, 0])
ax[1, 0].imshow(cube[:, :, 2, 0])
ax[1, 1].imshow(cube[:, :, 3, 0])
plt.savefig("test_map.png")
"""

fig1, ax1 = plt.subplots(2, 2)
ax1[0, 0].hist2d(x = ra[0, :], y = dec[0, :], weights = tod[0, 0, 0, :], bins = [120, 120])
ax1[0, 1].hist2d(x = ra[1, :], y = dec[1, :], weights = tod[1, 0, 0, :], bins = [120, 120])
ax1[1, 0].hist2d(x = ra[2, :], y = dec[2, :], weights = tod[2, 0, 0, :], bins = [120, 120])
ax1[0, 0].hist2d(x = ra[3, :], y = dec[3, :], weights = tod[3, 0, 0, :], bins = [120, 120])
plt.savefig("test_hist.png")
