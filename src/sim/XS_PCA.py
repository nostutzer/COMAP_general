import numpy as np
import matplotlib.pyplot as plt 
import h5py 
from tqdm import trange 
from sklearn.decomposition import PCA
import os 
import sys
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

fonts = {
    "font.family": "serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(fonts)

path = "/home/sagittarius/Documents/COMAP_general/COMAP_general/data/2D_xs/"

file_list = os.listdir(path)
files = []
for file in file_list:
    if "co6" in file:
        files.append(file)

print(files)

xs = np.zeros((len(files), 14, 14))
ks  = np.zeros((len(files), 2, 14))
edges_par = np.zeros((len(files), 15))
edges_perp = np.zeros((len(files), 15))

for i in trange(len(files)):
    name = path + files[i]
    with h5py.File(name, "r") as infile:
        spectrum = np.array(infile["xs_2D"])[()]
        k = np.array(infile["k"])[()]
        edge_par = np.array(infile["k_bin_edges_par"])[()]
        edge_perp = np.array(infile["k_bin_edges_perp"])[()]
        
        xs[i, ...]  = spectrum
        ks[i, ...]  = k
        edges_par[i, :] = edge_par
        edges_perp[i, :] = edge_perp

    infile.close()

xs = xs.reshape(xs.shape[0], 14 * 14)
print(np.all(xs == 0), xs.shape)

pca = PCA(25)

pca_data = pca.fit(xs)
print(pca.explained_variance_ratio_ * 100, 100 * np.sum(pca.explained_variance_ratio_))

def log2lin(x, k_edges):
    loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
    logx = np.log10(x) - np.log10(k_edges[0])
    return logx / loglen

# ax.set_xscale('log')
minorticks = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
                200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]

majorticks = [1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]
majorlabels = ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$']

xbins = edges_par[0, :]

ticklist_x = log2lin(minorticks, xbins)
majorlist_x = log2lin(majorticks, xbins)

ybins = edges_perp[0, :]

ticklist_y = log2lin(minorticks, ybins)
majorlist_y = log2lin(majorticks, ybins)


fig, ax = plt.subplots(5, 5, figsize = (10, 10), sharex = True, sharey = True)

cmap = cm.RdBu
vmin = -0.5
vmax = 0.5
clabels = list(np.arange(-0.4, 0.5, 0.2))
clabels = [round(i, 2) for i in clabels]
print(clabels)
for i in range(5):
    for j in range(5):
        comp = pca_data.components_[i * 5 + j, :].reshape(14, 14)
        img = ax[i, j].imshow(comp, interpolation = "none", origin = "lower",
                        cmap = cmap, vmin = vmin, vmax = vmax, extent = [0, 1, 0, 1], rasterized = True)
        
        
        ax[i, j].set_xticks(ticklist_x, minor=True)
        ax[i, j].set_xticks(majorlist_x, minor=False)
        ax[i, j].set_xticklabels(majorlabels, minor=False)
        ax[i, j].set_yticks(ticklist_y, minor=True)
        ax[i, j].set_yticks(majorlist_y, minor=False)
        ax[i, j].set_yticklabels(majorlabels, minor=False)

        ax[i, j].set_xlim(0, 1)
        ax[i, j].set_ylim(0, 1)
        ax[i, j].set_title(fr"PCA {pca.explained_variance_ratio_[i * 5 + j] * 100:.2g} %")
        divider = make_axes_locatable(ax[i, j])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = fig.colorbar(img, ax = ax[i, j], cax = cax)
        cbar.set_ticks(clabels)
        cbar.ax.set_yticklabels(clabels, rotation = 90)
        print(cbar.ax.get_yticklabels(), cbar.ax.get_xticklabels())

for i in range(5):        
    ax[-1, i].set_xlabel(r'$k_{\parallel}$ [Mpc$^{-1}$]')
    ax[i, 0].set_ylabel(r'$k_{\bot}$ [Mpc$^{-1}$]')
  
#fig.tight_layout()


plt.savefig("/home/sagittarius/Documents/COMAP_general/COMAP_general/figs/XS_PCA_co6.pdf", bbox_inches = "tight")



fig1, ax1 = plt.subplots(5, 5, figsize = (10, 10))

fig1.tight_layout()

spec = xs[1000, :].reshape(14, 14)

clabels = [-1e20, -1e10, 0, 1e10, 1e20]
clabel_text = ["$-10^{20}$", "$-10^{10}$", "$0$", "$10^{10}$", "$10^{20}$"]
print(clabels)

for i in range(5):
    for j in range(5):
        comp = pca_data.components_[i * 5 + j, :].reshape(14, 14)
        

        img = ax1[i, j].imshow(np.dot(comp, spec) * spec, interpolation = "none", origin = "lower",
                        cmap = cmap, vmin = None, vmax = None, extent = [0, 1, 0, 1], rasterized = True,
                        norm=colors.SymLogNorm(linthresh=0.01, linscale=0.01,
                                              vmin=-1e20, vmax=1e20))
        
        
        ax1[i, j].set_xticks(ticklist_x, minor=True)
        ax1[i, j].set_xticks(majorlist_x, minor=False)
        ax1[i, j].set_xticklabels(majorlabels, minor=False)
        ax1[i, j].set_yticks(ticklist_y, minor=True)
        ax1[i, j].set_yticks(majorlist_y, minor=False)
        ax1[i, j].set_yticklabels(majorlabels, minor=False)

        ax1[i, j].set_xlim(0, 1)
        ax1[i, j].set_ylim(0, 1)
        #ax1[i, j].set_title(fr"PCA {pca.explained_variance_ratio_[i * 5 + j] * 100:.2g} %")
        divider = make_axes_locatable(ax1[i, j])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = fig1.colorbar(img, ax = ax1[i, j], cax = cax)
        cbar.set_ticks(clabels)
        cbar.ax.set_yticklabels(clabel_text, rotation = 0)
        #print(cbar.ax.get_yticklabels(), cbar.ax.get_xticklabels())

for i in range(5):        
    ax1[-1, i].set_xlabel(r'$k_{\parallel}$ [Mpc$^{-1}$]')
    ax1[i, 0].set_ylabel(r'$k_{\bot}$ [Mpc$^{-1}$]')
  

fig2, ax2 = plt.subplots(1, figsize = (5, 5))
img = ax2.imshow(spec, interpolation = "none", origin = "lower",
                        cmap = cmap, vmin = None, vmax = None, extent = [0, 1, 0, 1], rasterized = True)
        

ax2.set_xticks(ticklist_x, minor=True)
ax2.set_xticks(majorlist_x, minor=False)
ax2.set_xticklabels(majorlabels, minor=False)
ax2.set_yticks(ticklist_y, minor=True)
ax2.set_yticks(majorlist_y, minor=False)
ax2.set_yticklabels(majorlabels, minor=False)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
#ax2.set_title(fr"PCA {pca.explained_variance_ratio_[i * 5 + j] * 100:.2g} %")
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size = "5%", pad = 0.05)
cbar = fig1.colorbar(img, ax = ax2, cax = cax)
#cbar.set_ticks(clabels)
#cbar.ax.set_yticklabels(clabels, rotation = 90)
plt.show()