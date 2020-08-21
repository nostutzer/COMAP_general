import numpy as np
import h5py
import matplotlib.pyplot as plt


path = "/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2020-05/"
filename = "comp_comap-0013798-2020-05-29-235130.hd5"

feed_idx = 0
freq_idx = 30
sb_idx = 1

f = h5py.File(path + filename, "r")
vane_angles    = np.array(f["/hk/antenna0/vane/angle"])/100.0  # Degrees
vane_time      = np.array(f["/hk/antenna0/vane/utc"])
array_features = np.array(f["/hk/array/frame/features"])
tod            = np.array(f["/spectrometer/tod"][feed_idx, sb_idx, freq_idx])
print(tod.shape)
tod_time       = np.array(f["/spectrometer/MJD"])
feeds = np.array(f["/spectrometer/feeds"])
vane_active    = array_features&(2**13) != 0

print(vane_active.shape, vane_time.shape)
vane_time1 = vane_time[:len(vane_active)//2]
vane_time2 = vane_time[len(vane_active)//2:]
vane_active1 = vane_active[:len(vane_active)//2]
vane_active2 = vane_active[len(vane_active)//2:]

if np.sum(vane_active1) > 5:
    vane_time1 = vane_time1[vane_active1]
    tod_start_idx = np.argmin(np.abs(vane_time1[0]-tod_time))
    tod_stop_idx = np.argmin(np.abs(vane_time1[-1]-tod_time))
    tod1 = tod[tod_start_idx : tod_stop_idx]
    tod_time1 = tod_time[tod_start_idx : tod_stop_idx]

    threshold_idxs = np.argwhere(tod1 > 0.9*np.max(tod1))
    min_idx1 = threshold_idxs[0][0] + 40
    max_idx1 = threshold_idxs[-1][0] - 40

    min_idx11 = max_idx1 - 25

    Phot1 = np.mean(tod1[min_idx1:max_idx1])
    t1 = (tod1[min_idx1] + tod1[max_idx1])/2.0
    print(Phot1)


if np.sum(vane_active2) > 5:
    vane_time2 = vane_time2[vane_active2]
    tod_start_idx = np.argmin(np.abs(vane_time2[0]-tod_time))
    tod_stop_idx = np.argmin(np.abs(vane_time2[-1]-tod_time))
    tod2 = tod[tod_start_idx : tod_stop_idx]
    tod_time2 = tod_time[tod_start_idx : tod_stop_idx]

    threshold_idxs = np.argwhere(tod2 > 0.9*np.nanmax(tod2))
    min_idx2 = threshold_idxs[0][0] + 40
    max_idx2 = threshold_idxs[-1][0] - 40

    Phot2 = np.mean(tod2[min_idx2:max_idx2])
    t2 = (tod2[min_idx2] + tod2[max_idx2])/2.0
    print(Phot2)


def Phot_interp(t):
    return (Phot1*(t2 - t) + Phot1*(t - t1))/(t2 - t1)

plt.figure(figsize=(14,10))
plt.axvline(x=tod_time1[min_idx1], ls="--", c="y")
plt.axvline(x=tod_time1[max_idx1], ls="--", c="y")
plt.axhline(y=Phot1, ls="--", c="y")
plt.plot(tod_time1, tod1, c="k")
plt.savefig("test1.png")

plt.figure(figsize=(14,10))
plt.axvline(x=tod_time2[min_idx2], ls="--", c="y")
plt.axvline(x=tod_time2[max_idx2], ls="--", c="y")
plt.axhline(y=Phot2, ls="--", c="y")
plt.plot(tod_time2, tod2, c="k")
plt.savefig("test2.png")

plt.figure(figsize=(14,10))
plt.plot(tod_time, tod, c="k")
plt.plot(tod_time, Phot_interp(tod_time), c="r")
plt.xlim(5.8998994e4+0.0005, 5.8998994e4+0.0007)
plt.savefig("test3.png")

