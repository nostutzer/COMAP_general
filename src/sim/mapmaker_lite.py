import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
import copy
import WCS
import time
import shutil
from tqdm import trange
import sys

t0 = time.time()

# L2
data_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/sim/wo_mask/co6/"

nside       = 120
histo = np.zeros((nside, nside))

dpix        = 2.0 / 60.0
fieldcent   = [226, 55]
nfeeds      = 18
nbin        = nside * nside

looplen = 0
for i in trange(2, 11, 1):
    filename   = data_path + f"co6_0015330{i:02d}.h5"
    infile      = h5py.File(filename, "r")
    tod         = np.array(infile["tod"]) 
    pointing    = np.array(infile["point_cel"]) 
    ra          = pointing[:, :, 0] 
    dec         = pointing[:, :, 1] 
    px_idx = np.zeros_like(dec, dtype = int)
    tod[:, 0, :, :] = tod[:, 0, ::-1, :]
    tod[:, 2, :, :] = tod[:, 2, ::-1, :]
    for j in range(nfeeds):  
        looplen += 1
        px_idx[j, :] = WCS.ang2pix([nside, nside], [-dpix, dpix], fieldcent, dec[j, :], ra[j, :])
        map, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (0, nside * nside), weights = tod[j, 2, 9, :]) 
        nhit, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (0, nside * nside))
        map /= nhit 
        map = np.nan_to_num(map, nan = 0)
        histo += map.reshape(nside, nside)     
    infile.close()

histo = np.where(histo != 0, histo * 1e6, np.nan) # Transforming K to muK

mapfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/sim/"
mapfile_name = mapfile_path + "co6_map.h5"
mapfile = h5py.File(mapfile_name, "r")
x = np.array(mapfile["x"])
y = np.array(mapfile["y"])

x_lim, y_lim = [None,None], [None,None]
dx = x[1] - x[0]
x_lim[0] = x[0] - 0.5*dx; x_lim[1] = x[-1] + 0.5*dx
dy = y[1] - y[0]
y_lim[0] = y[1] - 0.5*dy; y_lim[1] = y[-1] + 0.5*dy

fig, ax = plt.subplots(figsize=(10,6))

matplotlib.use("Agg")  # No idea what this is. It resolves an error when writing gif/mp4.
cmap_name = "CMRmap"
cmap = copy.copy(plt.get_cmap(cmap_name))

ax.set_ylabel('Declination [deg]')
ax.set_xlabel('Right Ascension [deg]')

aspect = dx/dy
img = ax.imshow(histo.T / looplen, extent=(x_lim[0],x_lim[1],y_lim[0],y_lim[1]), interpolation='nearest',
                    aspect=aspect, cmap=cmap, origin='lower',
                    vmin = -10000, vmax=10000)

ax.axhline(y[60], color = "g", alpha = 0.5, zorder = 2)
ax.axvline(x[60], color = "g", alpha = 0.5, zorder = 3)

#ax.set_title(title)
cbar = fig.colorbar(img)
cbar.set_label("$\mu K$")
plt.savefig("co6_l2_cross_map.png")

"""
# L1
data_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level1/2020-07/"
data_name = data_path + "comap-0015330-2020-07-31-040632_sim_norm.hd5"

nside       = 120
histo = np.zeros((nside, nside))
dpix        = 2.0 / 60.0
fieldcent   = [226, 55]
nfeeds      = 18
nbin        = nside * nside
print("Loading TOD: ", time.time() - t0, " sec"); t0 = time.time()
infile      = h5py.File(data_name, "r")
tod         = np.array(infile["spectrometer/tod"]) 
ra          = np.array(infile["spectrometer/pixel_pointing/pixel_ra"])
dec         = np.array(infile["spectrometer/pixel_pointing/pixel_dec"])
px_idx      = np.zeros_like(dec, dtype = int)
tod[:, 0, :, :] = tod[:, 0, ::-1, :]
tod[:, 2, :, :] = tod[:, 2, ::-1, :]
looplen = 0
print("Looping through feeds: ", time.time() - t0, " sec")
for j in trange(1):  
    looplen += 1
    px_idx[j, :] = WCS.ang2pix([nside, nside], [-dpix, dpix], fieldcent, dec[j, :], ra[j, :])
    map, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (0, nside * nside), weights = tod[j, 2, 9, :]) 
    nhit, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (0, nside * nside))
    map /= nhit 
    print(edges)
    map = np.nan_to_num(map, nan = 0)
    histo += map.reshape(nside, nside)     

histo = np.where(histo != 0, histo * 1e6, np.nan) # Transforming K to muK

mapfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/sim/"
mapfile_name = mapfile_path + "co6_map.h5"
mapfile = h5py.File(mapfile_name, "r")
x = np.array(mapfile["x"])
y = np.array(mapfile["y"])

X, Y = np.meshgrid(x, y, indexing = "ij")
X, Y = X.flatten(), Y.flatten()
pixvec = WCS.ang2pix([nside, nside], [-dpix, dpix], fieldcent, Y, X)
_px_idx = np.unique(px_idx[0, :])

for i in range(len(_px_idx)):
    print(_px_idx[i], pixvec[_px_idx[i]-1])
print(_px_idx)
print(pixvec)
print(np.max(pixvec), np.min(pixvec))

fig2, ax2 = plt.subplots()
ax2.plot(np.linspace(np.min(_px_idx), np.max(_px_idx), len(_px_idx)), _px_idx, label = "Pointing px")
ax2.plot(np.linspace(np.min(pixvec[_px_idx]), np.max(pixvec[_px_idx]), len(pixvec[_px_idx])), pixvec[_px_idx], label = "Grid px")
ax2.plot(np.linspace(np.min(pixvec[_px_idx - 1]), np.max(pixvec[_px_idx - 1]), len(pixvec[_px_idx - 1])), pixvec[_px_idx - 1], label = "Grid px - 1")
ax2.set_yscale("symlog")
ax2.legend()
#ax2.plot(np.arange(np.min(pixvec[_px_idx-1]), np.max(pixvec[_px_idx-1])), pixvec[_px_idx])
plt.savefig("test.png")
x_lim, y_lim = [None,None], [None,None]
dx = x[1] - x[0]
x_lim[0] = x[0] - 0.5*dx; x_lim[1] = x[-1] + 0.5*dx
dy = y[1] - y[0]
y_lim[0] = y[1] - 0.5*dy; y_lim[1] = y[-1] + 0.5*dy
print("lims: ", (x_lim[1] - x_lim[0]) / 120, (y_lim[1] - y_lim[0]) / 120)
fig1, ax1 = plt.subplots(figsize=(10,6))

matplotlib.use("Agg")  # No idea what this is. It resolves an error when writing gif/mp4.
cmap_name = "CMRmap"
cmap = copy.copy(plt.get_cmap(cmap_name))

ax1.set_ylabel('Declination [deg]')
ax1.set_xlabel('Right Ascension [deg]')

aspect = dx/dy
img1 = ax1.imshow(histo.T / looplen, extent=(x_lim[0],x_lim[1],y_lim[0],y_lim[1]), interpolation='nearest',
                    aspect=aspect, cmap=cmap, origin='lower',
                    vmin = -1e8, vmax=1e8)

#ax.set_title(title)
cbar = fig1.colorbar(img1)
cbar.set_label("$\mu K$")
plt.savefig("co6_l1_map.png")
"""
# Map
"""
mapfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/sim/wo_mask/"
mapfile_name = mapfile_path + "co6_map.h5"
mapfile = h5py.File(mapfile_name, "r")
x = np.array(mapfile["x"])
y = np.array(mapfile["y"])
map_coadd = np.array(mapfile["map_coadd"])
map_coadd = map_coadd[2, 9, :, :]

x_lim, y_lim = [None,None], [None,None]
dx = x[1] - x[0]
x_lim[0] = x[0] - 0.5*dx; x_lim[1] = x[-1] + 0.5*dx
dy = y[1] - y[0]
y_lim[0] = y[1] - 0.5*dy; y_lim[1] = y[-1] + 0.5*dy

fig3, ax3 = plt.subplots(figsize=(10,6))

matplotlib.use("Agg")  # No idea what this is. It resolves an error when writing gif/mp4.
cmap_name = "CMRmap"
cmap = copy.copy(plt.get_cmap(cmap_name))

ax3.set_ylabel('Declination [deg]')
ax3.set_xlabel('Right Ascension [deg]')

aspect = dx/dy

img = ax3.imshow(map_coadd * 1e6, extent=(x_lim[0],x_lim[1],y_lim[0],y_lim[1]), interpolation='nearest',
                    aspect=aspect, cmap=cmap, origin='lower',
                    vmin = -1e3, vmax=1e3, zorder = 1)

ax3.axhline(y[60], color = "g", alpha = 0.5, zorder = 2)
ax3.axvline(x[60], color = "g", alpha = 0.5, zorder = 3)


plt.savefig("co6_cross_overlay_map.png")
"""
