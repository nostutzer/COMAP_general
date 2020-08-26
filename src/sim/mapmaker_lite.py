import numpy as np
import h5py
import matplotlib.pyplot as plt
import WCS
import time
import shutil
from tqdm import trange
import sys

t0 = time.time()
"""
cube_path = "/mn/stornext/d16/cmbco/comap/protodir/"
cube_filename = cube_path + "cube_real.npy"
cube = np.load(cube_filename)
cubeshape = cube.shape
cube[::10, :, :] = np.max(cube)
cube[:, ::10, :] = np.max(cube)
cube = cube.reshape(cubeshape[0], cubeshape[0], 4, 1024)
"""
data_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/"
data_name = data_path + "tod_sim_15330.hd5"
print("Load infile: ", time.time() - t0, " sec")
infile  = h5py.File(data_name, "r")
tod     = np.array(infile["/spectrometer/tod"]) 
ra     = np.array(infile["/spectrometer/pixel_pointing/pixel_ra"]) 
dec     = np.array(infile["/spectrometer/pixel_pointing/pixel_dec"]) 
px_idx     = np.array(infile["/spectrometer/pixel_pointing/pixel_idx"]) 

print("Infile loaded: ", time.time() - t0, " sec")

"""
plt.figure()
plt.plot(np.arange(0, len(tod[5, 0, 0, :])), tod[5, 0, 0, :])
plt.plot(np.arange(0, len(tod[5, 1, 0, :])), tod[5, 1, 0, :])
plt.plot(np.arange(0, len(tod[5, 2, 0, :])), tod[5, 2, 0, :])
plt.plot(np.arange(0, len(tod[5, 3, 0, :])), tod[5, 3, 0, :])
plt.xlabel("RA")
plt.ylabel("Dec")
plt.savefig("test_tod_15330.png")
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(cube[:, :, 0, 0].T)
ax[0, 1].imshow(cube[:, :, 1, 0].T)
ax[1, 0].imshow(cube[:, :, 2, 0].T)
ax[1, 1].imshow(cube[:, :, 3, 0].T)
plt.savefig("test_map_15330.png")
"""
print("Computing histograms infile: ", time.time() - t0, " sec")
"""
fig1, ax1 = plt.subplots(2, 2)
hist1, edgeX1, edgeY1, im1 = plt.hist2d(x = ra[0, :], y = dec[0, :], weights = tod[0, 0, 125, :], bins = [120, 120])
nhit1, edgeX1, edgeY1, im1 = plt.hist2d(x = ra[0, :], y = dec[0, :], bins = [120, 120])
hist1 /= nhit1
ax1[0, 0].imshow(hist1.T)

hist2, edgeX2, edgeY2, im2 = plt.hist2d(x = ra[1, :], y = dec[1, :], weights = tod[1, 0, 125, :], bins = [120, 120])
nhit2, edgeX2, edgeY2, im2 = plt.hist2d(x = ra[1, :], y = dec[1, :], bins = [120, 120])
hist2 /= nhit2
ax1[0, 1].imshow(hist2.T)

hist3, edgeX3, edgeY3, im3 = plt.hist2d(x = ra[2, :], y = dec[2, :], weights = tod[2, 0, 125, :], bins = [120, 120])
nhit3, edgeX3, edgeY3, im3 = plt.hist2d(x = ra[2, :], y = dec[2, :], bins = [120, 120])
hist3 /= nhit3
ax1[1, 0].imshow(hist3.T)

hist4, edgeX4, edgeY4, im4 = plt.hist2d(x = ra[3, :], y = dec[3, :], weights = tod[3, 0, 125, :], bins = [120, 120])
nhit4, edgeX4, edgeY4, im4 = plt.hist2d(x = ra[3, :], y = dec[3, :], bins = [120, 120])
hist4 /= nhit4
ax1[1, 1].imshow(hist4.T)
plt.savefig("test_hist_15330.png")
"""
"""
fig1, ax1 = plt.subplots(2, 2)
hist1, edge1, im1 = plt.hist(px_idx[0, :], weights = tod[0, 0, 125, :], bins = 120 * 120)
nhit1, edge1, im1 = plt.hist(px_idx[0, :], bins = 120 * 120)
print(np.max(px_idx), np.min(px_idx))
print(hist1)
print(nhit1)
hist1 /= nhit1
hist1 = hist1.reshape(120, 120)
print("Hist shape: ", hist1.shape)
print("Hist reshape: ", hist1.reshape(120, 120).shape)
ax1[0, 0].imshow(hist1.T)

hist2, edge2, im2 = plt.hist(px_idx[1, :], weights = tod[1, 0, 125, :], bins = 120 * 120)
nhit2, edge2, im2 = plt.hist(px_idx[1, :], bins = 120 * 120)
hist2 /= nhit2
hist2 = hist2.reshape(120, 120)

ax1[0, 1].imshow(hist2.T)

hist3, edge3, im3 = plt.hist(px_idx[2, :], weights = tod[2, 0, 125, :], bins = 120 * 120)
nhit3, edge3, im3 = plt.hist(px_idx[2, :], bins = 120 * 120)
hist3 /= nhit3
hist3 = hist3.reshape(120, 120)

ax1[1, 0].imshow(hist3.T)

hist4, edge4, im4 = plt.hist(px_idx[3, :], weights = tod[3, 0, 125, :], bins = 120 * 120)
nhit4, edge4, im4 = plt.hist(px_idx[3, :], bins = 120 * 120)
hist4 /= nhit4
hist4 = hist4.reshape(120, 120)

ax1[1, 1].imshow(hist4.T)
"""
px = np.zeros(len(px_idx[0, :]) + 2)
time_stream = np.zeros(len(px_idx[0, :]) + 2)
px[0] = 0
px[-1] = 120 * 120
px[1: - 1] = px_idx[0, :]

time_stream[0] = 0
time_stream[-1] = 0
time_stream[1: - 1] = tod[0, 0, 125, :]

hist, edge, im = plt.hist(px, weights = time_stream, bins = 120 * 120)
nhit, edge, im = plt.hist(px, bins = 120 * 120)
print(np.max(px), np.min(px))
print(len(px_idx[0, :]), len(tod[0, 0, 125, :]))
hist /= nhit

print("Hist shape: ", hist.shape)
print("Hist reshape: ", hist.reshape(120, 120).shape)
hist = hist.reshape(120, 120)
string = "\n"
print(nhit)
print(edge)
#hist[np.isnan(hist)] == 0

print(np.sum(np.isnan(hist) == False))

fig, ax = plt.subplots()
ax.imshow(hist.T)

plt.savefig("test_hist_15330.png")