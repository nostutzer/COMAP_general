import numpy as np 
from scipy.sparse import csc_matrix, identity, diags
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sys
"""
N = 20
x = y = np.linspace(0, 1, N)
dx = dy = x[1] - x[0]

X, Y = np.meshgrid(x, y, indexing = "ij")


def franke_function(x1, x2):
    return .75 * np.exp(-(9 * x1 - 2) ** 2 / 4.0 - (9 * x2 - 2) ** 2 / 4.0) +\
    .75 * np.exp(-(9 * x1 + 1) ** 2 / 49.0 - (9 * x2 + 1) / 10.0) +\
    .5 * np.exp(-(9 * x1 - 7) ** 2 / 4.0 - (9 * x2 - 3) ** 2 / 4.0) -\
    .2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)

def scaning_strategy(t):
    return N / 2 * np.sin(2 * np.pi / 20 * t) + N/2, dy * t

m = franke_function(X, Y)


N_scan = 10

x = y = np.arange(0, 20, 20 / N)
dx = dy = x[1] - x[0]

X, Y = np.meshgrid(x, y, indexing = "ij")


x_flat, y_flat = X.flatten(), Y.flatten()
signal = m.flatten()

complete_scan = np.zeros((2, len(x_flat) * N_scan))
complete_signal = np.zeros(len(signal) * N_scan)

for i in range(N_scan):
    complete_scan[0, len(x_flat)*i:len(x_flat)*(i+1)] = x_flat
    complete_scan[1, len(y_flat)*i:len(y_flat)*(i+1)] = y_flat
    complete_signal[len(signal)*i:len(signal)*(i+1)]   = signal


"""


class Scanner():
    def __init__(self, N_pix, N_scan, N_basisperscan, norm = 1, COcube = True, cube_filename = None):
        self.norm = norm
        if COcube:
            self.cube_filename = cube_filename
            self.get_CO_map()
            self.N_pix = self.m.shape[0]
        else:
            self.N_pix  = N_pix
            self.signal_map()
        
        self.N_scan     = N_scan
        self.N_perscan  = self.N_pix ** 2
        self.N_tod      = self.N_perscan * self.N_scan    
        self.N_basisperscan    = N_basisperscan
        if self.N_basisperscan > self.N_perscan:
            print("N_basisperscan must be leq N_perscan")
            sys.exit()
        
    def franke_function(self, x1, x2):
        return .75 * np.exp(-(9 * x1 - 2) ** 2 / 4.0 - (9 * x2 - 2) ** 2 / 4.0) +\
        .75 * np.exp(-(9 * x1 + 1) ** 2 / 49.0 - (9 * x2 + 1) / 10.0) +\
        .5 * np.exp(-(9 * x1 - 7) ** 2 / 4.0 - (9 * x2 - 3) ** 2 / 4.0) -\
        .2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)

    def signal_map(self):
        N_pix = self.N_pix
        x = y = np.linspace(0, 1, N_pix)
        self.X, self.Y = np.meshgrid(x, y)
        self.m = self.franke_function(self.X, self.Y) * self.norm

    def get_CO_map(self):
        cube = np.load(self.cube_filename)
        cubeshape = cube.shape
        cube *= 1e-6 * self.norm    # Normalization of cube by input value
        cube = cube[:, :, 1201] 
        self.m = cube

    def simple_sequential_scan(self):
        N_scan = self.N_scan
        N_pix  = self.N_pix
        N_perscan = self.N_perscan
        x_scan = y_scan = np.arange(0, N_pix, 1)
        dx = dy = x_scan[1] - x_scan[0]

        X_scan, Y_scan = np.meshgrid(x_scan, y_scan)

        x_flat, y_flat = X_scan.flatten(), Y_scan.flatten()
        signal = self.m.flatten()
        complete_pxnr = np.tile(np.arange(N_perscan), N_scan)
        complete_scan = np.zeros((2, self.N_tod))
        complete_signal = np.zeros(self.N_tod)

        complete_scan[0, :] = x_flat[complete_pxnr]
        complete_scan[1, :] = y_flat[complete_pxnr]
        complete_signal  = signal[complete_pxnr]

        self.signal = complete_signal 
        self.x      = complete_scan[0, :]
        self.y      = complete_scan[1, :]
        self.px     = complete_pxnr
        
    def random_scan(self):
        N_tod  = self.N_tod
        N_scan = self.N_scan
        N_pix  = self.N_pix
        N_perscan = self.N_perscan
        
        x_scan = y_scan = np.arange(0, N_pix, 1)
        X_scan, Y_scan = np.meshgrid(x_scan, y_scan)
        x_flat, y_flat = X_scan.flatten(), Y_scan.flatten()

        signal = self.m.flatten()
        np.random.seed(666)
        complete_pxnr = np.random.randint(N_perscan, size = N_tod)
        complete_scan = np.zeros((2, self.N_tod))
        complete_signal = np.zeros(N_tod)
        
        complete_scan[0, :] = x_flat[complete_pxnr]
        complete_scan[1, :] = y_flat[complete_pxnr]
        complete_signal  = signal[complete_pxnr]
        
        """
        complete_pxnr = np.random.randint(N_perscan, size = N_perscan)
        complete_scan = np.zeros((2, self.N_tod))
        complete_signal = np.zeros(N_tod)
        
        complete_scan[0, :] = np.tile(x_flat[complete_pxnr], N_scan)
        complete_scan[1, :] = np.tile(y_flat[complete_pxnr], N_scan)
        complete_signal  = np.tile(signal[complete_pxnr], N_scan)
        complete_pxnr    = np.tile(complete_pxnr, N_scan)
        """
        self.signal = complete_signal 
        self.x      = complete_scan[0, :]
        self.y      = complete_scan[1, :]
        self.px     = complete_pxnr
        
    def P_tod2map(self):
        N_tod, N_scan, N_perscan = self.N_tod, self.N_scan, self.N_perscan
        """
        N_t = self.N_perscan
        P = np.zeros((N_t, N_t * self.N_scan))
        for i in range(self.N_scan):
            P[:, i * N_t:(i + 1) * N_t] = np.identity(N_t)
        """
        ones = np.ones(N_tod)
        cols = np.arange(0, N_tod, 1)
        rows = self.px         
        P = csc_matrix((ones, (rows, cols)), shape = (N_perscan, N_tod))
        return P     

    def P_map2tod(self):
        N_tod, N_scan, N_perscan = self.N_tod, self.N_scan, self.N_perscan
        """
        P = np.zeros((N_t * self.N_scan, N_t))
        for i in range(self.N_scan):
            P[i * N_t:(i + 1) * N_t, :] = np.identity(N_t)
        """
        ones = np.ones(N_tod)
        rows = np.arange(0, N_tod, 1)
        cols = self.px        
        #cols = np.tile(cols, N_scan) 
        P = csc_matrix((ones, (rows, cols)), shape = (N_tod, N_perscan))
        return P 
    
    def F_scan2tod(self):
        N_tod, N_scan, N_perscan, N_basisperscan = self.N_tod, self.N_scan, self.N_perscan, self.N_basisperscan
        """
        N_perscan = self.N_perscan
        F = np.zeros((self.N_tod, self.N_scan))
        for i in range(self.N_scan):
            F[i * N_perscan:(i + 1) * N_perscan, i] = 1
        """
        ones = np.ones(N_tod)
        rows = np.arange(0, N_tod, 1)
        cols = np.zeros(N_tod)
        basis_len = int(N_perscan / N_basisperscan)
        for i in range(N_scan * N_basisperscan):
            cols[i * basis_len:(i+1) * basis_len] = np.tile(i, basis_len)
        F = csc_matrix((ones, (rows, cols)), shape = (N_tod, N_scan * N_basisperscan))
        return F 

    def C_n(self, sigma0):
        N_tod, N_scan, N_perscan = self.N_tod, self.N_scan, self.N_perscan
        C_n = np.zeros(N_tod)
        for i in range(N_scan):
            C_n[i * N_perscan:(i+1)*N_perscan] = np.tile(sigma0[i], N_perscan)
        rows = cols = np.arange(0, N_tod, 1)
        C_n = csc_matrix((C_n, (rows, cols)), shape = (N_tod, N_tod))
        return C_n 

    def C_n_inv(self, sigma0):
        N_tod, N_scan, N_perscan = self.N_tod, self.N_scan, self.N_perscan
        C_n = np.zeros(N_tod)
        for i in range(N_scan):
            C_n[i * N_perscan:(i+1)*N_perscan] = np.tile(1 / sigma0[i], N_perscan)
        rows = cols = np.arange(0, N_tod, 1)
        C_n_inv = csc_matrix((C_n, (rows, cols)), shape = (N_tod, N_tod))
        return C_n_inv 


    def add_one_over_f_noise(self, tod):
        N_pix, N_tod, N_scan, N_perscan = self.N_pix, self.N_tod, self.N_scan, self.N_perscan
        y_noise = np.zeros(N_tod)
        samplerate = 10
        freq = np.fft.rfftfreq(N_perscan * 2) * samplerate
        sigma0_arr = np.zeros(N_scan)
        ft_tot = np.zeros((N_scan, N_perscan))
        np.random.seed(42)
        for i in range(N_scan):
            sigma0 = np.random.normal(2, 0.5, 1)
            sigma0_arr[i] = sigma0
            fknee = np.max((np.random.normal(1e-2, 1e-3), 1e-4))
            alpha = np.random.normal(2, 0.2)
            Ps = sigma0 ** 2 * (1 + (fknee/freq) ** alpha)
            Ps[0] = 0
            ft = np.zeros(N_perscan)
            for j in range(N_perscan):
                ft[j] = np.random.normal(0, Ps[j])
                ft_tot[i, j] = ft[j]
            #noise = np.fft.irfft(ft)[(N_perscan) // 2:(3 * N_perscan) // 2]
            noise = np.fft.irfft(ft)[:N_perscan]
            noise += y[i*N_perscan - 1] - noise[0]
            y_noise[i * N_perscan:(i+1)*N_perscan] += noise
        tod += y_noise
        
        return tod, sigma0_arr, y_noise

    def get_PCP_inv(self, P, C_n_inv):
        PC = P.transpose().dot(C_n_inv)        
        PCP = PC.dot(P)
        return linalg.inv(PCP + diags(1e-8*np.ones(self.N_perscan)))

    def get_destriped_map(self, P, C_n_inv, F, tod):
        PCP_inv = self.get_PCP_inv(P, C_n_inv)
        Z = identity(self.N_tod) - P.dot(PCP_inv).dot(P.transpose()).dot(C_n_inv)
        A = F.transpose().dot(C_n_inv).dot(Z).dot(F)
        b = F.transpose().dot(C_n_inv).dot(Z).dot(tod)
        self.A = A
        print(A)
        self.a = linalg.spsolve(A, b)
        
        m = PCP_inv.dot(P.transpose()).dot(C_n_inv).dot(tod - F.dot(self.a))
        return m.reshape(self.N_pix, self.N_pix)

    def get_bin_averaged_map(self, P, tod):
        A = P.transpose().dot(P)
        b = P.transpose().dot(tod)
        return linalg.spsolve(A, b).reshape(self.N_pix, self.N_pix)
        
    def get_noise_weighted_map(self, P, C_n_inv, tod):
        PCP_inv = self.get_PCP_inv(P, C_n_inv)
        m = PCP_inv.dot(P.transpose()).dot(C_n_inv).dot(tod)
        return m.reshape(self.N_pix, self.N_pix)

N_scan = 100
N_pix  = 120
N_perscan = N_pix ** 2
cube_filename = "/mn/stornext/d16/cmbco/comap/protodir/cube_real.npy"
scn = Scanner(N_pix, N_scan, 1, 3000, COcube = True, cube_filename = cube_filename)

m = scn.m

#scn.simple_sequential_scan()
scn.random_scan()
signal, x, y, px = scn.signal, scn.x, scn.y, scn.px
P_tod2map = scn.P_tod2map()
P = scn.P_map2tod()
F = scn.F_scan2tod()

tod, sigma0, y_noise = scn.add_one_over_f_noise(signal)
C_n = scn.C_n(sigma0)
C_n_inv = scn.C_n_inv(sigma0)
#print(P_tod2map.toarray())
#print(P_map2tod.toarray().shape)
#print(C_n_inv.toarray().shape)

tod -= np.nanmean(tod)
tod_normscan = tod.copy()

for i in range(N_scan):
    tod_normscan[i * N_perscan:(i+1)*N_perscan] -= np.nanmean(tod[i * N_perscan:(i+1)*N_perscan])

t = time.time()
m_destr = scn.get_destriped_map(P, C_n_inv, F, tod_normscan)
print(time.time() - t, "sec")
m_avg = scn.get_bin_averaged_map(P, tod_normscan)
m_weighted = scn.get_noise_weighted_map(P, C_n_inv, tod_normscan)

fig5, ax5 = plt.subplots(figsize = (5, 5)) 
ax5 = plt.imshow(scn.A.toarray(), rasterized = True)
plt.savefig("test0.png")

noise_ft = np.fft.rfft(y_noise)

tod_after = tod_normscan - F.dot(scn.a)

#scn.simple_sequential_scan()
scn.random_scan()
signal, x, y, px = scn.signal, scn.x, scn.y, scn.px

chi2_destr = np.sum((m_destr.T - m.T) ** 2)
chi2_weighted = np.sum((m_weighted.T - m.T) ** 2)
chi2_avg = np.sum((m_avg.T - m.T) ** 2)
print(chi2_destr, chi2_weighted, chi2_avg)

fig, ax = plt.subplots(2, 4, figsize = (10, 5))
im0 = ax[0, 0].imshow(m.T, vmin = -0.05, vmax = 0.05)
im1 = ax[0, 1].imshow(m_destr.T, vmin = -0.05, vmax = 0.05)
im2 = ax[0, 2].imshow(m_weighted.T, vmin = -0.05, vmax = 0.05)
im3 = ax[0, 3].imshow(m_avg.T, vmin = -0.05, vmax = 0.05)

#im4 = ax[1, 0].imshow(m.T - m.T, vmin = -0.02, vmax = 0.02)
im5 = ax[1, 1].imshow(m_destr.T - m.T, vmin = -0.02, vmax = 0.02)
im6 = ax[1, 2].imshow(m_weighted.T - m.T, vmin = -0.02, vmax = 0.02)
im7 = ax[1, 3].imshow(m_avg.T - m.T, vmin = -0.02, vmax = 0.02)

ax[0, 0].set_title("Original")
ax[0, 1].set_title("Destriped")
ax[0, 2].set_title("Noise Weighted")
ax[0, 3].set_title("Bin averaged")

ax[1, 1].set_title("Residual\n" + fr"$\chi^2 = ${chi2_destr:.3f}")
ax[1, 2].set_title("Residual\n" + fr"$\chi^2 = ${chi2_weighted:.3f}")
ax[1, 3].set_title("Residual\n" + fr"$\chi^2 = ${chi2_avg:.3f}")

divider0 = make_axes_locatable(ax[0, 0])
divider1 = make_axes_locatable(ax[0, 1])
divider2 = make_axes_locatable(ax[0, 2])
divider3 = make_axes_locatable(ax[0, 3])

#divider4 = make_axes_locatable(ax[1, 0])
divider5 = make_axes_locatable(ax[1, 1])
divider6 = make_axes_locatable(ax[1, 2])
divider7 = make_axes_locatable(ax[1, 3])

cax0 = divider0.append_axes("right", size="5%", pad=0.05)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cax3 = divider3.append_axes("right", size="5%", pad=0.05)

#cax4 = divider4.append_axes("right", size="5%", pad=0.05)
cax5 = divider5.append_axes("right", size="5%", pad=0.05)
cax6 = divider6.append_axes("right", size="5%", pad=0.05)
cax7 = divider7.append_axes("right", size="5%", pad=0.05)

fig.colorbar(im0, ax=ax[0, 0], cax = cax0)
fig.colorbar(im1, ax=ax[0, 1], cax = cax1)
fig.colorbar(im2, ax=ax[0, 2], cax = cax2)
fig.colorbar(im3, ax=ax[0, 3], cax = cax3)

#fig.colorbar(im4, ax=ax[1, 0], cax = cax4)
fig.colorbar(im5, ax=ax[1, 1], cax = cax5)
fig.colorbar(im6, ax=ax[1, 2], cax = cax6)
fig.colorbar(im7, ax=ax[1, 3], cax = cax7)

fig.tight_layout()
plt.savefig("test3_1basis.pdf")

fig1, ax1 = plt.subplots(3, 1, figsize = (10, 15))
ax1[0].plot(np.arange(len(x[:200])), x[:200], label = "x")
ax1[0].plot(np.arange(len(y[:200])), y[:200], label = "y")
ax1[0].legend()

ax1[1].plot(np.arange(len(tod_normscan)), tod_normscan, label = "TOD")
ax1[1].plot(np.arange(len(tod_normscan)), tod_after, label = "TOD - Fa", alpha = 0.5)
ax1[1].plot(np.arange(len(tod_normscan)), F.dot(scn.a), label = "Fa")
ax1[1].legend()


ax1[2].plot(np.arange(len(noise_ft)), np.abs(noise_ft) ** 2, label = "ft noise")
ax1[2].legend()
ax1[2].set_yscale("symlog")
ax1[2].set_xscale("log")

ax1[0].set_xlabel("Time [Arbitrary Units]")
ax1[0].set_ylabel("Distance [Arbitrary Units]")

ax1[1].set_xlabel("Time [Arbitrary Units]")
ax1[1].set_ylabel("TOD [Arbitrary Units]")

ax1[0].set_xlabel("Time [Arbitrary Units]")
ax1[0].set_ylabel("Noise Power Spectral Density [Arbitrary Units]")


fig1.tight_layout()
plt.savefig("test4_1basis.pdf")
plt.show()


