import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.fft as fft
import time
import glob
import os
import h5py
import WCS
from shutil import copyfile

try:
    filename = sys.argv[1]
except IndexError:
    print('Missing filename!')
    print('Usage: python cube2tod.py l1_filename.hd5')
    sys.exit(1)

with h5py.File(filename, mode="r") as fd:

    att = fd['comap'].attrs
    obs_id = int(att['obsid'])

sim_l1_name = 'sims/my_sim_%06i.hd5' % obs_id

copyfile(filename, sim_l1_name)

#cubefile = 'Li2015_N4096_0000.hd5'

tsys = 55

# with h5py.File(cubefile, mode="r") as my_file:
#     cube = np.array(my_file['cube'])[:]
#     sh = cube.shape
#     cube = cube.reshape(sh[0], 4, 1024)
#     cube = cube.transpose(1, 2, 0)
#     #cube = cube * 50000
#     cube = np.exp(cube / cube.flatten().std()) / 5

cube = np.load('cube_real.npy')
# print(cube.shape)
# plt.imshow(cube[:, :, 0], interpolation='none') #np.mean(cube[:,:,:16], 2), interpolation='none')
# plt.colorbar()
# plt.show()
# sys.exit()
my_map = cube
sh = cube.shape
cube = cube.reshape(sh[0] * sh[1], 4, 1024)
cube = cube.transpose(1, 2, 0)

#print(cube.flatten().std())
cube = cube * 1e-6 * 10000

print(cube.shape)
fieldcent = [226.0, 55.0]
with h5py.File(sim_l1_name, mode="r+") as my_file:
    ra = np.array(my_file[u'spectrometer/pixel_pointing/pixel_ra'][:])
    dec = np.array(my_file[u'spectrometer/pixel_pointing/pixel_dec'][:])
    nside = 120
    dpix = 2.0 / 60

    tod = my_file[u'spectrometer/tod']
    n_feed = len(tod[:, 0, 0, 0])
    for i in range(n_feed):
        print(i + 1)
        pixvec = WCS.ang2pix([nside, nside], [-dpix, dpix], [226.0, 55.0], dec[i, :], ra[i, :])
        # for j in range(len(pixvec)):
            # tod[i,:,:,j] *= (1.0 + cube[:, :, pixvec[j]] / 60.0)
        tod[i,:,:,:] += np.nanmean(np.array(tod[i,:,:,:]), 2)[:,:,None] * cube[:, :, pixvec] / tsys

sh = my_map.shape
my_map = my_map.reshape(sh[0], sh[1], 4, 64, 16).mean(4)
my_map = my_map.transpose(2, 3, 1, 0)
# plt.figure()
# plt.imshow(my_map[2, :, :, 0].T, interpolation='none')
# plt.show()
rms = np.zeros_like(my_map) + 1.0

#sh = my_map.shape
#my_map = np.random.randn(*sh)

outname = 'sim_map.h5'
f2 = h5py.File(outname, 'w')

dra = nside * dpix / 2 / np.cos(fieldcent[1] * np.pi / 180)
ra = np.linspace(-dra, dra, nside + 1) + fieldcent[0]
ra = 0.5 * (ra[1:] + ra[:-1])
ddec = nside * dpix / 2 
dec = np.linspace(-ddec, ddec, nside + 1) + fieldcent[1]
dec = 0.5 * (dec[1:] + dec[:-1])
f2.create_dataset('x', data=ra)
f2.create_dataset('y', data=dec)
f2.create_dataset('n_x', data=nside)
f2.create_dataset('n_y', data=nside)
f2.create_dataset('map_beam', data=my_map)
f2.create_dataset('rms_beam', data=rms)
f2.close()


# n_samp = 100
# ra = np.linspace(169, 171, n_samp)
# dec = np.linspace(52, 53, n_samp)


# dec2 = dec[:, None] + 0 * ra[None, :]
# dec2 = dec2.flatten()

# ra = ra[None, :] + 0 * dec[:, None]
# ra = ra.flatten()
# dec = dec2
# print(pixvec.shape)
# my_map = np.zeros_like(cube)
# n_map = np.zeros_like(my_map)

# for i in range(len(pixvec)):
#     my_map[pixvec[i]] += cube[pixvec[i]]
#     n_map[pixvec[i]] += 1.0
# my_map = my_map / n_map

# cube = cube.reshape(128, 128, 4096)
# my_map = my_map.reshape(128, 128, 4096)
# plt.imshow(cube[:,:,0], interpolation='none')
# plt.colorbar()
# plt.figure()
# plt.imshow(my_map[:,:,0], interpolation='none')
# plt.colorbar()
# plt.show()
