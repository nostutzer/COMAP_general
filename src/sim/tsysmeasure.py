import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import trange
import time
import ctypes

class TsysMeasure:
    def __init__(self):
        pass

    def load_data_from_arrays(self, vane_angles, vane_times, array_features, T_hot, tod, tod_times):
        self.vane_angles = vane_angles
        self.vane_times = np.array(vane_times, dtype=np.float64)
        self.array_features = array_features
        self.Thot = np.array(T_hot/100.0 + 273.15, dtype=np.float64)
        self.tod = np.array(tod, dtype=np.float32)
        self.tod_times = np.array(tod_times, dtype=np.float64)
        self.nr_vane_times = len(vane_times)

        vane_active = array_features&(2**13) != 0
        self.vane_time1 = vane_times[:self.nr_vane_times//2]
        self.vane_time2 = vane_times[self.nr_vane_times//2:]
        self.vane_active1 = vane_active[:self.nr_vane_times//2]
        self.vane_active2 = vane_active[self.nr_vane_times//2:]

        self.nfeeds, self.nbands, self.nfreqs, self.ntod = tod.shape

        self.Pcold = tod

        self.Phot = np.zeros((self.nfeeds, self.nbands, self.nfreqs, 2), dtype=np.float64)  # P_hot measurements from beginning and end of obsid.
        self.Phot_t = np.zeros((self.nfeeds, self.nbands, self.nfreqs, 2), dtype=np.float64)
        self.Phot[:] = np.nan  # All failed calcuations of Tsys should result in a nan, not a zero.
        self.Phot_t[:] = np.nan

        self.points_used = np.zeros((self.nfeeds, self.nbands, self.nfreqs))

        self.TCMB = 2.725


    def load_data_from_file(self, filename):
        f = h5py.File(filename, "r")
        vane_angles    = np.array(f["/hk/antenna0/vane/angle"])/100.0  # Degrees
        vane_times     = np.array(f["/hk/antenna0/vane/utc"])
        array_features = np.array(f["/hk/array/frame/features"])
        tod            = np.array(f["/spectrometer/tod"])#[feed_idx, sb_idx, freq_idx])
        # tod = tod.reshape(1,1,1,tod.shape[0])
        tod_times      = np.array(f["/spectrometer/MJD"])
        feeds          = np.array(f["/spectrometer/feeds"])
        if tod_times[0] > 58712.03706:
            T_hot      = np.array(f["/hk/antenna0/vane/Tvane"])
        else:
            T_hot      = np.array(f["/hk/antenna0/env/ambientLoadTemp"])
        self.load_data_from_arrays(vane_angles, vane_times, array_features, T_hot, tod, tod_times)


    def solve(self):
        ### Step 1: Calculate P_hot at the start and end Tsys measurement points. ###
        vane_time1, vane_time2, vane_active1, vane_active2, tod, tod_times = self.vane_time1, self.vane_time2, self.vane_active1, self.vane_active2, self.tod, self.tod_times
        nfeeds, nbands, nfreqs, ntod = self.nfeeds, self.nbands, self.nfreqs, self.ntod
        for i, vane_timei, vane_activei in [[0, vane_time1, vane_active1], [1, vane_time2, vane_active2]]:
            if np.sum(vane_activei) > 5:  # If Tsys
                vane_timei = vane_timei[vane_activei]
                tod_start_idx = np.argmin(np.abs(vane_timei[0]-tod_times))
                tod_stop_idx = np.argmin(np.abs(vane_timei[-1]-tod_times))
                self.calib_indices_tod[i, :] = tod_start_idx, tod_stop_idx
                for feed_idx in range(nfeeds):
                    for band_idx in range(nbands):
                        for freq_idx in range(nfreqs):
                            todi = tod[feed_idx, band_idx, freq_idx, tod_start_idx : tod_stop_idx]
                            tod_timesi = tod_times[tod_start_idx : tod_stop_idx]
                            if np.sum(todi > 0) > 10:  # Check number of valid points. Also catches NaNs.
                                threshold_idxs = np.argwhere(todi > 0.95*np.max(todi))  # Points where tod is at least 95% of max. (We assume this is only true during Tsys measurement).
                                min_idxi = threshold_idxs[0][0] + 40  # Take the first and last of points fulfilling the above condition, assume they represent start and end of 
                                max_idxi = threshold_idxs[-1][0] - 40  # Tsys measurement, and add a 40-idx safety margin (I think this is 40*20ms = approx 1 second.)
                                if max_idxi > min_idxi:
                                    self.Phot[feed_idx, band_idx, freq_idx, i] = np.nanmean(todi[min_idxi:max_idxi])
                                    self.Phot_t[feed_idx, band_idx, freq_idx, i] = (tod_timesi[min_idxi] + tod_timesi[max_idxi])/2.0
                                    self.points_used[feed_idx, band_idx, freq_idx] = max_idxi - min_idxi

        ### Step 2: Interpolate P_hot onto all tod time points ###

        # self.Phot = np.zeros((nfeeds, nbands, nfreqs, self.ntod), dtype=np.float32)  # P_hot linearly interpolated onto all vane timestamps.
        # for feed_idx in trange(nfeeds):
        #     for band_idx in range(nbands):
        #         for freq_idx in range(nfreqs):
        #             self.Phot[feed_idx, band_idx, freq_idx, :] = Phot_interp_func(feed_idx, band_idx, freq_idx, self.tod_times)
        
        ### Step 3: Inteprolate T_hot onto all tod time points. ### 
        self.Thot_interp_func = interp1d(self.vane_times, self.Thot, kind="nearest", copy=False, bounds_error=False)


        # self.Thot = np.zeros((self.ntod), dtype=np.int32)
        # for i in trange(self.ntod):
        #     self.Thot[i] = self.Thot[np.argmin(np.abs(self.vane_times - tod_times[i]))]

        ### Step 4: Calculate Tsys and interpolate it. ###
        # self.Tsys = (self.Thot - self.TCMB)/(self.Phot/self.Pcold - 1)
        # self.Tsys = np.array(self.Tsys, dtype=np.float32)
        # self.Tsys_interp = interp1d(self.tod_times, self.Tsys, copy=False)
    # def Phot_interp_func(self, t):
    #     Ph1, Ph2 = self.Phot[:,:,:,:,None]
    #     t1, t2 = self.Phot_t[:,:,:,:,None]
    #     return (Ph1*(t2 - t[None,None,None,:]) + Ph2*(t[None,None,None,:] - t1))/(t2 - t1)

    def Tsys_of_t(self, t, Pcold):
        t0 = time.time()
        self.Thot = self.Thot_interp_func(t)
        print("Thot: ",time.time()-t0)
        t0 = time.time()

        self.Tsys = np.zeros((self.nfeeds, self.nbands, self.nfreqs, self.ntod), dtype=np.float32)
        tsyslib = ctypes.cdll.LoadLibrary("/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/tsyslib.so.1")
        float64_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags="contiguous")
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")
        float64_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=4, flags="contiguous")

        tsyslib.tsys_calc.argtypes = [float32_array4, float32_array4, float64_array1,
                                    float64_array1, float64_array4, float64_array4,
                                    ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        tsyslib.tsys_calc(self.Tsys, self.tod, self.Thot, t, self.Phot, self.Phot_t, self.TCMB,
                        self.nfeeds, self.nbands, self.nfreqs, self.ntod)
        # self.Tsys = self.Thot - self.TCMB
        # self.Tsys /= (self.Phot_interp_func(t)/self.Pcold - 1)
        print("Ph/Pc: ",time.time()-t0)
        return self.Tsys
        # return (self.Thot_interp_func(t) - self.TCMB)/(self.Phot_interp_func(t)/self.Pcold - 1)

    # def Tsys_of_t(self, t):
    #     return self.Tsys_interp(t)



if __name__ == "__main__":
    tod_in_path = "/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2020-05/"
    tod_in_filename = tod_in_path + "comp_comap-0013518-2020-05-20-110000.hd5"

    Tsys = TsysMeasure()
    Tsys.load_data_from_file(tod_in_filename)
    Tsys.solve()

    tsys = Tsys.Tsys_of_t(Tsys.tod_times, Tsys.tod)

    # np.save("tsys2.npy", tsys)

    # np.save("tsys_interp.npy", Tsys.Tsys_of_t(Tsys.vane_times))
    # np.save("tsys.npy", Tsys.Tsys)
    # np.save("thot.npy", Tsys.Thot)
    # np.save("phot.npy", Tsys.Phot)
    # np.save("pcold.npy", Tsys.Pcold)
    # np.save("points_used.npy", Tsys.points_used)