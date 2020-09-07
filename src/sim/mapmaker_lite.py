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


# L1
data_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level1/2020-07/"
data_name = data_path + "comap-0015330-2020-07-31-040632_sim_cross.hd5"
"""
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
for j in trange(nfeeds):  
    looplen += 1
    px_idx[j, :] = WCS.ang2pix([nside, nside], [-dpix, dpix], fieldcent, dec[j, :], ra[j, :])
    map, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (0, nside * nside), weights = tod[j, 2, 9, :]) 
    nhit, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (0, nside * nside))
    print(px_idx[j, :])
    print(nhit[nhit != 0])
    map /= nhit 
    print(edges)
    map = np.nan_to_num(map, nan = 0)
    histo += map.reshape(nside, nside)     

histo = np.where(histo != 0, histo / np.nanmax(histo) * 1e6, np.nan) # Transforming K to muK

mapfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/sim/"
mapfile_name = mapfile_path + "co6_map.h5"
mapfile = h5py.File(mapfile_name, "r")
x = np.array(mapfile["x"])
y = np.array(mapfile["y"])

X, Y = np.meshgrid(x, y, indexing = "ij")
X, Y = X.flatten(), Y.flatten()
pixvec = WCS.ang2pix([nside, nside], [-dpix, dpix], fieldcent, Y, X)
_px_idx = np.unique(px_idx[0, :])

x_lim, y_lim = [None,None], [None,None]
dx = x[1] - x[0]
x_lim[0] = x[0] - 0.5*dpix; x_lim[1] = x[-1] + 0.5*dpix
dy = y[1] - y[0]
y_lim[0] = y[1] - 0.5*dpix; y_lim[1] = y[-1] + 0.5*dpix
print("lims: ", (x_lim[1] - x_lim[0]) / 120, (y_lim[1] - y_lim[0]) / 120)
fig1, ax1 = plt.subplots(figsize=(10,6))

matplotlib.use("Agg")  # No idea what this is. It resolves an error when writing gif/mp4.
cmap_name = "CMRmap"
cmap = copy.copy(plt.get_cmap(cmap_name))

ax1.set_ylabel('Declination [deg]')
ax1.set_xlabel('Right Ascension [deg]')

aspect = dx/dy
img1 = ax1.imshow(histo.T / looplen, extent=(x_lim[0],x_lim[1],y_lim[0],y_lim[1]), interpolation='nearest',
                    aspect=aspect, cmap=cmap, origin='lower')#,
                    #vmin = -1e30, vmax=1e30)
histo_L1 = histo / looplen
#ax.set_title(title)
cbar = fig1.colorbar(img1)
cbar.set_label("$\mu K$")
plt.savefig("co6_l1_map.png")
#plt.show()
print("L1 plotted:")
print(np.nanmin(histo / looplen))
print(np.nanmax(histo / looplen))
"""
# L2

mapfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/sim/"
mapfile_name = mapfile_path + "co6_map.h5"
mapfile = h5py.File(mapfile_name, "r")
x = np.array(mapfile["x"])
y = np.array(mapfile["y"])

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
    ra          = pointing[:, :, 0] #- (x[1] - x[0])
    dec         = pointing[:, :, 1] #- (y[1] - y[0])
    px_idx = np.zeros_like(dec, dtype = int)
    tod[:, 0, :, :] = tod[:, 0, ::-1, :]
    tod[:, 2, :, :] = tod[:, 2, ::-1, :]
    for j in range(nfeeds):  
        looplen += 1
        px_idx[j, :] = WCS.ang2pix([nside, nside], [-dpix, dpix], fieldcent, dec[j, :], ra[j, :])
        map, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (0, nside * nside), weights = tod[j, 2, 9, :]) 
        nhit, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (0, nside * nside))
        #map, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (-0.5, nside * nside - 0.5), weights = tod[j, 2, 9, :]) 
        #nhit, edges   = np.histogram(px_idx[j, :], bins = nside * nside, range = (-0.5, nside * nside - 0.5))
        map /= nhit 
        map = np.nan_to_num(map, nan = 0)
        histo += map.reshape(nside, nside)     
    infile.close()

histo = np.where(histo != 0, histo * 1e6, np.nan) # Transforming K to muK
x_lim, y_lim = [None,None], [None,None]
dx = x[1] - x[0]
x_lim[0] = x[0] - 0.5*dx; 
x_lim[1] = x[-1] + 0.5*dx
dy = y[1] - y[0]
y_lim[0] = y[1] - 0.5*dy; 
y_lim[1] = y[-1] + 0.5*dy
"""
_histo = np.zeros((nside, nside))
print(_histo[:-1, :-1].shape, histo[1:, 1:].shape)
_histo[:-1, :-1] = histo[1:, 1:]
histo = _histo
"""

fig, ax = plt.subplots(figsize=(10,6))

cmap_name = "CMRmap"
cmap = copy.copy(plt.get_cmap(cmap_name))

ax.set_title("After l2gen")
ax.set_ylabel('Declination [deg]')
ax.set_xlabel('Right Ascension [deg]')

aspect = dx/dy

img = ax.imshow(histo.T / looplen, extent=(x_lim[0],x_lim[1],y_lim[0],y_lim[1]), interpolation='nearest',
                    aspect=aspect, cmap=cmap, origin='lower',
                    vmin = -8000, vmax=8000)

#ax.axhline(y[60], color = "g", alpha = 0.5, zorder = 2)
#ax.axvline(x[60], color = "g", alpha = 0.5, zorder = 3)

#ax.set_title(title)
cbar = fig.colorbar(img)
cbar.set_label("$\mu K$")
plt.savefig("co6_l2_cross_map.png")


# Map

mapfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/sim/wo_mask/"
mapfile_name = mapfile_path + "co6_map.h5"
mapfile = h5py.File(mapfile_name, "r")
x = np.array(mapfile["x"])
y = np.array(mapfile["y"])
map_coadd = np.array(mapfile["map_coadd"])
map_coadd = map_coadd[2, 9, :, :]

print("X: ", x[59])
print("Y: ", y[59])
x_lim, y_lim = [None,None], [None,None]
dx = x[1] - x[0]
x_lim[0] = x[0] - 0.5*dx; x_lim[1] = x[-1] + 0.5*dx
dy = y[1] - y[0]
y_lim[0] = y[1] - 0.5*dy; y_lim[1] = y[-1] + 0.5*dy

fig3, ax3 = plt.subplots(figsize=(10,6))

cmap_name3 = "CMRmap"
cmap3 = copy.copy(plt.get_cmap(cmap_name3))

ax3.set_ylabel('Declination [deg]')
ax3.set_xlabel('Right Ascension [deg]')

ax3.set_title("After l2gen + tod2comap")
aspect = dx/dy
map_coadd = np.where(map_coadd != 0, map_coadd * 1e6, np.nan) # Transforming K to muK

#map_coadd[:58, :] = np.nan
map_coadd[59, :]  = -2 * np.nanmax(map_coadd) 
#map_coadd[61:, :] = np.nan
#map_coadd[:, :60] = np.nan
map_coadd[:, 59]  = -2 * np.nanmax(map_coadd) 
#map_coadd[:, 61:] = np.nan

img3 = ax3.imshow(map_coadd, extent=(x_lim[0],x_lim[1],y_lim[0],y_lim[1]), interpolation='nearest',
                    aspect=aspect, cmap=cmap3, origin='lower',
                    vmin = -1e4, vmax=1e4, zorder = 1)

#ax3.axhline(y[60], color = "g", alpha = 0.5, zorder = 2)
#ax3.axvline(x[60], color = "g", alpha = 0.5, zorder = 3)

cbar3 = fig3.colorbar(img3)
cbar3.set_label("$\mu K$")
plt.savefig("co6_cross_overlay_map2.png")

v_idx = np.arange(0, 120)
h_idx = v_idx.copy()

v_selec = np.arange(0, 120, 3)
h_selec = v_selec.copy()

dec_v, ra_v = WCS.pix2ang([nside, nside], [-dpix, dpix], fieldcent, np.zeros_like(v_idx), v_idx)
dec_h, ra_h = WCS.pix2ang([nside, nside], [-dpix, dpix], fieldcent, h_idx, np.zeros_like(h_idx))


w = WCS.Info2WCS([nside, nside], [-dpix, dpix], fieldcent)
dec_1D, ra_1D = WCS.pix2ang1D(w, [nside, nside], v_idx)

pix_v = WCS.ang2pix([nside, nside], [-dpix, dpix], fieldcent, dec_v, ra_v)
pix_v_original = np.ones_like(v_idx) * 120 + v_idx
#print(pix_v)
#print(pix_v_original)

hist_v = histo[v_idx, :] / looplen
hist_v = hist_v.T / np.nanmax(np.absolute(hist_v))

map_v  = map_coadd[v_idx, :]
map_v  = map_v.T / np.nanmax(np.absolute(map_v))

hist_h = histo[:, h_idx] / looplen
hist_h = hist_h / np.nanmax(np.absolute(hist_h))

map_h  = map_coadd[:, h_idx]
map_h  = map_h / np.nanmax(np.absolute(map_h))

fig4, ax4 = plt.subplots(2, 1)

ax4[0].plot(dec_v, hist_v, color = "r", drawstyle='steps-mid', zorder = 2, linewidth = 1)

ax5 = ax4[0].twiny()
ax5.plot(y, map_v, color = "g", drawstyle='steps-mid', zorder = 1, linewidth = 1)
ax5.tick_params(axis='x', labelcolor="g")
ax5.set_xlim(np.min(dec_v), np.max(dec_v))
ax5.set_xlabel("Dec")
ax5.legend(["map"], loc = 2)
ax4[0].set_ylabel("Normalized by maximum")
ax4[0].legend(["l2"], loc = 1)
ax4[0].set_xlim(np.min(dec_v), np.max(dec_v))
ax4[0].tick_params(axis='x', labelcolor="r")
ax4[0].set_xlabel("Dec")

ax4[1].plot(ra_h, hist_h, color = "r", drawstyle='steps-mid', zorder = 2, linewidth = 1)

ax6 = ax4[1].twiny()
ax6.plot(x, map_h, color = "g", drawstyle='steps-mid', zorder = 1, linewidth = 1)
ax6.tick_params(axis='x', labelcolor="g")
ax6.set_xlim(np.min(ra_h), np.max(ra_h))
ax6.set_xlabel("RA")
ax6.legend(["map"], loc = 2)

ax4[1].set_xlabel("index")
ax4[1].set_ylabel("Normalized by maximum")
#ax4[1].legend(["l2", "map"])
ax4[1].tick_params(axis='x', labelcolor="r")
ax4[1].set_xlabel("RA")
ax4[1].set_xlim(np.min(ra_h), np.max(ra_h))
ax4[1].legend(["l2"], loc = 1)
"""
import scipy.signal as signal

print(signal.argrelmax(hist_h, axis = 0, order = 30))
i, j = signal.argrelmax(hist, axis = 0, order = 30)
print(i.shape, j.shape, hist_h[i].shape, ra_h[j].shape)
ax4[1].plot(ra_h[i], hist_h[j], "bo", markersize = 1)
"""
fig4.tight_layout()
plt.savefig("slice.png")


plt.show()