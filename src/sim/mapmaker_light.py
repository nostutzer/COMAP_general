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
import argparse
import os
import re
import ctypes


class MapMakerLight():
    def __init__(self):
        self.nside  = 120
        self.dpix   = 2.0 / 60.0
        self.nbin   = self.nside ** 2 
        
        self.input()

    def input(self):
        """
        Function parsing the command line input.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--param", type = str, default = None,
                            help = """Full path and name to input parameter file.""")
        parser.add_argument("-o", "--outfile", type = str, default = None,
                            help = """Name of output file (without file ending).""")

        parser.add_argument("-l", "--level", type = int, default = None,
                            help = """Whether the input/output file are of level1 or level2.""")
        
        args = parser.parse_args()

        if args.param == None:
            message = """No input parameter file given, please provide an input parameter file!"""
            raise NameError(message)
        elif args.outfile == None:
            message = """No output file given, please provide an input file (without file ending)!"""
            raise NameError(message)
        elif args.level == None or args.level not in (1, 2):
            message = """The level of the input data is not provided! E.g. -l 1 or -l 2 (only level1 and level2 supported!)"""
            raise NameError(message)
        else:
            self.param_file     = args.param
            self.outfile    = args.outfile
            self.level      = args.level

    def read_paramfile(self):
        """
        Function reading the parameter file provided by the command line
        argument, and defining class parameters.
        """
        param_file  = open(self.param_file, "r")
        params      = param_file.read()

        runlist_path = re.search(r"\nRUNLIST\s*=\s*'(\/.*?)'", params)  # Defining regex pattern to search for runlist path in parameter file.
        self.runlist_path = str(runlist_path.group(1))                  # Extracting path
        
        l1_in_path = re.search(r"\nTOD_OUT_DIR\s*=\s*'(\/.*?)'", params)    # Defining regex pattern to search for level1 file path.
        self.l1_in_path = str(l1_in_path.group(1))                        # Extracting path
    
        l2_in_path = re.search(r"\nLEVEL2_DIR\s*=\s*'(\/.*?)'", params)    # Defining regex pattern to search for level1 file path.
        self.l2_in_path = str(l2_in_path.group(1))                        # Extracting path
    
        l1_out_path = re.search(r"\nL1_MAP_DIR\s*=\s*'(\/.*?)'", params)   # Defining regex pattern to search for directory where to put the level1 maps.
        self.l1_out_path = str(l1_out_path.group(1))                          # Extracting path
        
        l2_out_path = re.search(r"\nL2_MAP_DIR\s*=\s*'(\/.*?)'", params)   # Defining regex pattern to search for directory where to put the level2 maps.
        self.l2_out_path = str(l2_out_path.group(1))                          # Extracting path

        runlist_file = open(self.runlist_path, "r")         # Opening 
        runlist = runlist_file.read()
        tod_in_list = re.findall(r"\/.*?\.\w+", runlist)
        self.tod_in_list = tod_in_list
        
        tod_in_list = re.findall(r"\/.*?\.\w+", runlist)
        obsIDs_list = re.findall(r"\s\d{6}\s", runlist)
        self.obsIDs_list = obsIDs_list
        self.nobsIDs_list = len(self.obsIDs_list)
        
        patch_name = re.search(r"\s([a-zA-Z0-9]+)\s", runlist)
        self.patch_name = str(patch_name.group(1))


        patch_def_path = re.search(r"\nPATCH_DEFINITION_FILE\s*=\s*'(\/.*?)'", params)
        self.patch_def_path = str(patch_def_path.group(1))

        patch_def_file = open(self.patch_def_path, "r")
        patch_def = patch_def_file.read()
        fieldcent   = re.search(rf"{self.patch_name}\s*([0-9.]+)\s*([0-9.]+)", patch_def) 
        self.fieldcent = [eval(fieldcent.group(1)), eval(fieldcent.group(2))]

        runlist_file.close()
        param_file.close()

        print("Patch def:", self.patch_def_path)
        print("Patch", self.patch_name)
        print("Field center", self.fieldcent)
        print("Runlist:", self.runlist_path)
        print("L1 in:", self.l1_in_path)
        print("L1 out:", self.l1_out_path)
        print("L2 in:", self.l2_in_path)
        print("L2 out:", self.l2_out_path)
        print("# obsID", len(self.tod_in_list))
        print("obsID file #1: ", self.tod_in_list[0])
        print("obsID #1: ", self.obsIDs_list[0])
        
    def run(self):
        self.read_paramfile()
        if self.level == 1:
            print("Loopig through runlist L1: ")
            for i in trange(len(self.tod_in_list)):
                self.infile    = self.l1_in_path + self.tod_in_list[i]
                self.outfile   = self.outfile + "_map.h5"
                        
                self.readL1()
                self.make_mapL1()
                self.write_mapL1()

            print("Through loop L1")
        else:
            print("Loopig through runlist L2: ")
            for i in trange(len(self.tod_in_list)):
                self.obsID = self.obsIDs_list[i]
                l2_files = []
                for filename in os.listdir(self.l2_in_path):
                    if f"{self.obsID}" in filename:
                        l2_files.append(filename)
                self.l2_files = l2_files
                self.make_mapL2()
                self.write_mapL2()
            print("Through loop L2")

    def readL1(self):
        t = time.time()
        infile          = h5py.File(self.infile, "r")
        self.tod        = np.array(infile["spectrometer/tod"])[()].astype(dtype=np.float64, copy=False) 
        self.ra         = np.array(infile["spectrometer/pixel_pointing/pixel_ra"])[()]
        self.dec        = np.array(infile["spectrometer/pixel_pointing/pixel_dec"])[()]
        
        self.tod[:, 0, :, :] = self.tod[:, 0, ::-1, :]
        self.tod[:, 2, :, :] = self.tod[:, 2, ::-1, :]
        self.nfeeds, self.nsb, self.nfreq, self.nsamp = self.tod.shape
        infile.close()
        print("L1 file read: ", time.time() - t, "sec")
    
    def readL2(self):
        print("Reading L2 file:")
        infile      = h5py.File(self.infile, "r")
        self.tod    = np.array(infile["tod"])[()].astype(dtype=np.float32, copy=False) 
        pointing    = np.array(infile["point_cel"])[()]
        self.ra     = pointing[:, :, 0] 
        self.dec    = pointing[:, :, 1] 

        self.tod[:, 0, :, :] = self.tod[:, 0, ::-1, :]
        self.tod[:, 2, :, :] = self.tod[:, 2, ::-1, :]
        self.nfeeds, self.nsb, self.nfreq, self.nsamp = self.tod.shape
        infile.close()
        
    def hist(self, idx, tod):
        map = np.zeros((self.nsb, self.nfreq, self.nbin), dtype = ctypes.c_double)
        nhit = np.zeros((self.nsb, self.nfreq, self.nbin), dtype = ctypes.c_int)

        maputilslib = ctypes.cdll.LoadLibrary("histutils.so.1")  # Load shared C utils library.
        float32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=3, flags="contiguous")   # 4D array 32-bit float pointer object.
        float64_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=3, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=3, flags="contiguous")       # 4D array 32-bit integer pointer object.
        int32_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, flags="contiguous")       # 4D array 32-bit integer pointer object.

        maputilslib.histogram.argtypes = [int32_array1, float64_array3, float64_array3, int32_array3,        # Specifying input types for C library function.
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
        maputilslib.histogram(idx, tod, map, nhit, self.nsb, self.nfreq, self.nsamp, self.nbin)
        return map, nhit

    def make_mapL1(self):
        print("Making L1 map:") 
        histo = np.zeros((self.nsb, self.nfreq, self.nside, self.nside))    
        allhits = np.zeros((self.nsb, self.nfreq, self.nside, self.nside))    
        px_idx      = np.zeros_like(self.dec, dtype = ctypes.c_int)
        looplen = 0
                
        t = time.time()
        for i in trange(self.nfeeds):  
            looplen += 1
            px_idx[i, :] = WCS.ang2pix([self.nside, self.nside], 
                                        [-self.dpix, self.dpix], 
                                        self.fieldcent, 
                                        self.dec[i, :], 
                                        self.ra[i, :])
            map, nhit = self.hist(px_idx[i, :], self.tod[i, ...])
            hit, edge = np.histogram(px_idx[i, :], bins = self.nbin, range = (0, self.nbin))
            npmap, edge = np.histogram(px_idx[i, :], bins = self.nbin, range = (0, self.nbin), weights = self.tod[i, 3, 125, :])
            print(nhit.shape, hit.shape)
            """for i in range(self.nsamp):
                print(hit[i], nhit[2, 9, i])
            """
            for i in range(120 * 120):
                print(np.absolute(map[3, 125, i] - npmap[i]) / npmap[i], map[3, 125, i], npmap[i])
            print(np.allclose(hit, nhit[3, 125, :]))
            print(np.allclose(npmap, map[3, 125, :]))
            sys.exit()
            #map     = np.apply_along_axis(self.hist, axis = -1, arr = px_idx[i, :], args = self.tod[i, ...])
            #nhit    = np.apply_along_axis(self.hist, axis = -1, arr = px_idx[i, :], args = None)
            histo += map.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
            allhits += nhit.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
        print("\nLoop time: ", time.time() - t, " sec\n")

        histo      /= allhits
        histo       = np.nan_to_num(histo, nan = 0)
        self.map    = histo / looplen
        self.nhit   = allhits
        print("DOne with mapmaking:")

    def make_mapL2(self):
        print("Makign L2 map:") 
        histo   = np.zeros((self.nsb, self.nfreq, self.nside, self.nside))
        px_idx  = np.zeros_like(self.dec, dtype = int)
        looplen = 0
        for i in trange(len(self.l2_files)):
            self.filename   = self.l2_in_path + l2_files[i]
            self.readL2()
            for j in range(self.nfeeds):  
                looplen += 1
                px_idx[j, :] = WCS.ang2pix([self.nside, self.nside], 
                                            [-self.dpix, self.dpix], 
                                            self.fieldcent, 
                                            self.dec[j, :], 
                                            self.ra[j, :])

                map     = np.apply_along_axis(self.hist, axis = -1, arr = px_idx[j, :], args = self.tod[j, ...])
                nhit    = np.apply_along_axis(self.hist, axis = -1, arr = px_idx[j, :], args = None)

                histo += map.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
                allhits += nhit.reshape(self.nsb, self.nfreq, self.nside, self.nside)    
        
        histo      /= allhits
        histo       = np.nan_to_num(histo, nan = 0)
        self.map    = histo / looplen
        self.nhit   = allhits        

    def write_mapL1(self):
        print("Not yet done! Writing L1 map to file:")

    def write_mapL2(self):
        print("Not yet done! Writing L1 map to file:")

if __name__ == "__main__":
    maker = MapMakerLight()
    #maker.read_paramfile()
    maker.run()
    
    """
    t0 = time.time()

    # L1
    data_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level1/sim/staticTsys/2020-07/"
    data_name = data_path + "comap-0015326-2020-07-31-001728.hd5"
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
    plt.savefig("co6_015326_l1_map.png")
    #plt.show()
    print("L1 plotted:")
    print(np.nanmin(histo / looplen))
    print(np.nanmax(histo / looplen))
    """
    # L2
    """

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
    for i in trange(2, 5, 1):
        filename   = data_path + f"co6_0015326{i:d}.h5"
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
    plt.savefig("co6_015326_l2_map.png")
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
    #map_coadd[59, :]  = -2 * np.nanmax(map_coadd) 
    #map_coadd[61:, :] = np.nan
    #map_coadd[:, :60] = np.nan
    #map_coadd[:, 59]  = -2 * np.nanmax(map_coadd) 
    #map_coadd[:, 61:] = np.nan

    img3 = ax3.imshow(map_coadd, extent=(x_lim[0],x_lim[1],y_lim[0],y_lim[1]), interpolation='nearest',
                        aspect=aspect, cmap=cmap3, origin='lower',
                        vmin = -1e4, vmax=1e4, zorder = 1)

    #ax3.axhline(y[60], color = "g", alpha = 0.5, zorder = 2)
    #ax3.axvline(x[60], color = "g", alpha = 0.5, zorder = 3)

    cbar3 = fig3.colorbar(img3)
    cbar3.set_label("$\mu K$")
    plt.savefig("co6_015326_final_map.png")

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
    import scipy.signal as signal

    print(signal.argrelmax(hist_h, axis = 0, order = 30))
    i, j = signal.argrelmax(hist, axis = 0, order = 30)
    print(i.shape, j.shape, hist_h[i].shape, ra_h[j].shape)
    ax4[1].plot(ra_h[i], hist_h[j], "bo", markersize = 1)

    fig4.tight_layout()
    plt.savefig("slice.png")
    plt.show()
    """