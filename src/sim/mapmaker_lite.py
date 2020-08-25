import numpy as np
import h5py
import matplotlib.pyplot as plt
import WCS
import time
import shutil
from tqdm import trange
import sys

t0 = time.time()

cube_path = "/mn/stornext/d16/cmbco/comap/protodir/"
cube_filename = cube_path + "cube_real.npy"
cube = np.load(cube_filename)
cubeshape = cube.shape
cube = cube.reshape(cubeshape[0], cubeshape[0], 4, 1024)

data_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/"
data_name = data_path + "tod_sim.hd5"
print("Load infile: ", time.time() - t0, " sec")
infile  = h5py.File(data_name, "r")
tod     = np.array(infile["/spectrometer/tod"]) 

ra     = np.array(infile["/spectrometer/pixel_pointing/pixel_ra"]) 
dec     = np.array(infile["/spectrometer/pixel_pointing/pixel_dec"]) 
print("Infile loaded: ", time.time() - t0, " sec")

"""
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(cube[:, :, 0, 0].T)
ax[0, 1].imshow(cube[:, :, 1, 0].T)
ax[1, 0].imshow(cube[:, :, 2, 0].T)
ax[1, 1].imshow(cube[:, :, 3, 0].T)
plt.savefig("test_map.png")
"""
print("Computing histograms infile: ", time.time() - t0, " sec")

fig1, ax1 = plt.subplots(2, 2)
hist1, edgeX1, edgeY1, im1 = plt.hist2d(x = ra[0, :], y = dec[0, :], weights = np.sum(tod[:, 0, 500, :], axis = 0), bins = [120, 120])
nhit1, edgeX1, edgeY1, im1 = plt.hist2d(x = ra[0, :], y = dec[0, :], bins = [120, 120])
hist1 /= nhit1
ax1[0, 0].imshow(hist1.T)

hist2, edgeX2, edgeY2, im2 = plt.hist2d(x = ra[1, :], y = dec[1, :], weights = np.sum(tod[:, 1, 500, :], axis = 0), bins = [120, 120])
nhit2, edgeX2, edgeY2, im2 = plt.hist2d(x = ra[1, :], y = dec[1, :], bins = [120, 120])
hist2 /= nhit2
ax1[0, 1].imshow(hist2.T)

hist3, edgeX3, edgeY3, im3 = plt.hist2d(x = ra[2, :], y = dec[2, :], weights = np.sum(tod[:, 2, 500, :], axis = 0), bins = [120, 120])
nhit3, edgeX3, edgeY3, im3 = plt.hist2d(x = ra[2, :], y = dec[2, :], bins = [120, 120])
hist3 /= nhit3
ax1[1, 0].imshow(hist3.T)

hist4, edgeX4, edgeY4, im4 = plt.hist2d(x = ra[3, :], y = dec[3, :], weights = np.sum(tod[:, 3, 500, :], axis = 0), bins = [120, 120])
nhit4, edgeX4, edgeY4, im4 = plt.hist2d(x = ra[3, :], y = dec[3, :], bins = [120, 120])
hist4 /= nhit4
ax1[1, 1].imshow(hist4.T)

fig2, ax2 = plt.subplots(1, 2)
ax2[0].imshow(hist1.T)
ax2[1].imshow(hist4.T)
print(hist1)
print(np.all(hist4))
print(np.all(nhit4))
plt.savefig("test_hist.png")
