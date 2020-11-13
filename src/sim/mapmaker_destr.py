import numpy as np 
from scipy.sparse import csc_matrix, identity, diags
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sys
import h5py
import WCS
import copy


class Destriper():
    def __init__(self, nsb, nfreq, nfeed, infile, N_baseline, highcut, eps):
        self.nsb         = nsb 
        self.nfreq       = nfreq 
        self.nfeed       = nfeed
        self.infile      = infile
        self.N_baseline  = N_baseline
        self.highcut     = highcut
        self.baseline_time = 1 / highcut
        self.eps         = eps

        self.Nside       = 120           # Number of pixels along RA/Dec
        self.dpix        = 2.0 / 60.0    # Pixel resolution in degrees (2' = 2/60 deg)
        self.Npix        = self.Nside ** 2   # Total number of pixels in the image
        self.fieldcent = [226, 55]  

    def run(self):
        print("Loading data:")
        t0 = time.time()
        self.get_data()
        print("Loading time:", time.time() - t0, "sec")

        print("Get pixel index:")
        t0 = time.time()
        self.get_px_index()
        print("Get pixel time:", time.time() - t0, "sec")

        print("Get pointing matrix:")
        t0 = time.time()
        self.get_P()
        print("Get pointing matrix time:", time.time() - t0, "sec")

        print("Get F:")
        t0 = time.time()
        self.get_F()
        print("Get F time:", time.time() - t0, "sec")
        
        print("Get destriped map:")
        t0 = time.time()
        self.get_destriped_map()
        print("Get destriped map time:", time.time() - t0, "sec")
        
    def get_data(self):
        nfeed, nsb, nfreq = self.nfeed, self.nsb, self.nfreq
        infile      = h5py.File(self.infile, "r")
        self.tod    = np.array(infile["tod"])[nfeed, nsb, nfreq, :] #[()]#.astype(dtype=np.float32, copy=False) 
        if self.tod.dtype != np.float32:
            raise ValueError("The input TOD should be of dtype float32!")
        
        self.time   = np.array(infile["time"])[()] * 3600 * 24
        self.sigma0 = np.array(infile["sigma0"])[nfeed, nsb, nfreq] #[()]

        pointing    = np.array(infile["point_cel"])[nfeed, :, :] #[()]
        self.ra     = pointing[:, 0] 
        self.dec    = pointing[:, 1] 

        #self.tod[:, 0, :, :] = self.tod[:, 0, ::-1, :]
        #self.tod[:, 2, :, :] = self.tod[:, 2, ::-1, :]
        #self.Nfeeds, self.Nsb, self.Nfreq, self.Nsamp = self.tod.shape
        self.Nsamp = self.tod.shape[0]
        self.dt    = self.time[1] - self.time[0]
        self.N_baseline = int(round((self.time[-1] - self.time[0]) / self.baseline_time))
        self.N_perbaseline = int(np.floor(self.Nsamp / self.N_baseline))
        
        self.Nsamp = self.N_perbaseline * self.N_baseline
        
        self.tod  = self.tod[:self.Nsamp] - np.nanmean(self.tod[:self.Nsamp])
        self.time = self.time[:self.Nsamp]
        self.ra   = self.ra[:self.Nsamp]
        self.dec  = self.dec[:self.Nsamp]

        print(len(self.tod)) 
        print(self.N_baseline, self.time[-1] - self.time[0], self.dt * self.N_baseline, np.floor(self.Nsamp / self.N_baseline))
        infile.close()

    def get_px_index(self):
        Nside, dpix, fieldcent, ra, dec = self.Nside, self.dpix, self.fieldcent, self.ra, self.dec

        self.px = WCS.ang2pix([Nside, Nside], [-dpix, dpix], fieldcent, dec, ra)     
    
    def get_P(self):
        Nsamp, Npix = self.Nsamp, self.Npix

        ones = np.ones(Nsamp)
        rows = np.arange(0, Nsamp, 1)
        cols = self.px        
        #cols = np.tile(cols, N_scan) 
        self.P = csc_matrix((ones, (rows, cols)), shape = (Nsamp, Npix))
        

    def get_F(self):
        Nsamp, N_baseline, N_perbaseline = self.Nsamp, self.N_baseline, self.N_perbaseline
        
        ones = np.ones(Nsamp)
        rows = np.arange(0, Nsamp, 1)
        cols = np.zeros(Nsamp)
        for i in range(N_baseline):
            cols[i * N_perbaseline:(i+1) * N_perbaseline] = np.tile(i, N_perbaseline)
        self.F = csc_matrix((ones, (rows, cols)), shape = (Nsamp, N_baseline))
        
    def get_Cn_inv(self):
        Nsamp = self.Nsamp
        C_n_inv = np.ones(Nsamp) * 1 / self.sigma0
        self.C_n_inv = diags(C_n_inv)

    def get_destriped_map(self):
        self.get_P()
        self.get_F()
        self.get_Cn_inv()
        P, C_n_inv, F, tod, Nsamp, eps = self.P, self.C_n_inv, self.F, self.tod, self.Nsamp, self.eps
        
        PCP_inv = P.transpose().dot(C_n_inv)        
        PCP_inv = PCP_inv.dot(P) + diags(eps * np.ones(self.Npix))
        PCP_inv = diags(1 / PCP_inv.diagonal())

        Z = identity(Nsamp) - P.dot(PCP_inv).dot(P.transpose()).dot(C_n_inv)
        A = F.transpose().dot(C_n_inv).dot(Z).dot(F)
        b = F.transpose().dot(C_n_inv).dot(Z).dot(tod)
        
        self.a = linalg.spsolve(A, b)
        
        m = PCP_inv.dot(P.transpose()).dot(C_n_inv).dot(tod - F.dot(self.a))
        self.m = m.reshape(self.Nside, self.Nside)

    def get_bin_averaged_map(self):
        self.get_P()
        P, tod, eps = self.P, self.tod, self.eps
        A = P.transpose().dot(P) + diags(eps * np.ones(self.Npix))
        b = P.transpose().dot(tod)
        self.m = linalg.spsolve(A, b).reshape(self.Nside, self.Nside)
    
    def get_noise_weighted_map(self):
        self.get_P()
        self.get_Cn_inv()
        
        P, C_n_inv, tod, Nsamp, eps = self.P, self.C_n_inv, self.tod, self.Nsamp, self.eps 
        
        PCP_inv = P.transpose().dot(C_n_inv)        
        PCP_inv = PCP_inv.dot(P) + diags(eps * np.ones(self.Npix))
        PCP_inv = diags(1 / PCP_inv.diagonal())
        
        self.m = PCP_inv.dot(P.transpose()).dot(C_n_inv).dot(tod).reshape(self.Nside, self.Nside)
    
    def get_hits(self):
        self.get_P()
        P, Nside, Nsamp = self.P, self.Nside, self.Nsamp
        ones = np.ones(Nsamp)
        self.hits = P.transpose().dot(ones).reshape(Nside, Nside)

if __name__ == "__main__":
    nsb     = 3
    nfreq   = 30
    nfeed   = 5
    #infile  = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/sim/dynamicTsys/co6/co6_001498708.h5"
    #infile  = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/sim/dynamicTsys/co6/co6_001527505.h5"
    infile  = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/sim/dynamicTsys/co6/co6_001532602.h5"
    
    N_baseline = 10
    highcut = 0.02 #Hz 
    eps     = 1e-8

    t = time.time()

    destr = Destriper(nsb, nfreq, nfeed, infile, N_baseline, highcut, eps)
    destr.run()
    print("Total time elapsed:", time.time() - t, "sec")

    m_destr = destr.m 

    destr.get_noise_weighted_map()
    m_weighted = destr.m 
    
    destr.get_bin_averaged_map()
    m_avg = destr.m 

    destr.get_hits()
    hits = destr.hits
    """
    m_destr     = np.ma.masked_where(hits < 1, m_destr)
    m_weighted  = np.ma.masked_where(hits < 1, m_weighted)
    m_avg       = np.ma.masked_where(hits < 1, m_avg)
    hits        = np.ma.masked_where(hits < 1, hits)
    """
    cmap_name = "CMRmap"
    cmap = copy.copy(plt.get_cmap(cmap_name))
    cmap.set_bad("0.8", 1)

    print(np.sum(hits > 0))
    print(np.sum(hits > 1))
    print(np.sum(hits > -1))
    fig, ax = plt.subplots(1, 4, figsize = (10, 5))
    im3 = ax[3].imshow(hits.T, cmap = cmap)
    hits[hits <= 0] = np.nan
    
    im0 = ax[0].imshow(m_destr.T, cmap = cmap)
    im1 = ax[1].imshow(m_weighted.T, cmap = cmap)
    #im2 = ax[2].imshow(m_avg.T, cmap = cmap)
    im2 = ax[2].imshow(hits.T, cmap = cmap)

    im0.set_rasterized(True)
    im1.set_rasterized(True)
    im2.set_rasterized(True)
    im3.set_rasterized(True)

    divider0 = make_axes_locatable(ax[0])
    divider1 = make_axes_locatable(ax[1])
    divider2 = make_axes_locatable(ax[2])
    divider3 = make_axes_locatable(ax[3])

    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im0, ax=ax[0], cax = cax0)
    fig.colorbar(im1, ax=ax[1], cax = cax1)
    fig.colorbar(im2, ax=ax[2], cax = cax2)
    fig.colorbar(im3, ax=ax[3], cax = cax3)



    plt.savefig("test0.pdf")
    
