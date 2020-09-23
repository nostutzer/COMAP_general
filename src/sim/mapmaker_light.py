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
        """
        Initializing MapMakerLight class and defining class attributes.
        """
        self.nside  = 120           # Number of pixels along RA/Dec
        self.dpix   = 2.0 / 60.0    # Pixel resolution in degrees (2' = 2/60 deg)
        self.nbin   = self.nside ** 2   # Total number of pixels in the image
        self.template_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/templates/" # Path to map template
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

        parser.add_argument("-l", "--level", type = str, default = None,
                            help = """Whether the input/output file are of level1 or level2.""")
        
        parser.add_argument("-n", "--norm", type = float, default = 1.0,
                            help = """Normalize simulation cube by input value.""")
        
        parser.add_argument("-r", "--rms", action = "store_false",
                            help = """Set simulation map's rms to one.""")
        
        args = parser.parse_args()

        if args.param == None:
            message = """No input parameter file given, please provide an input parameter file!"""
            raise NameError(message)
        elif args.outfile == None:
            message = """No output file given, please provide an input file (without file ending)!"""
            raise NameError(message)
        elif args.level == None or args.level not in ("1", "2", "cube"):
            message = """The level of the input data is not provided! E.g. -l 1, -l 2 or -l cube (only level1, level2 and datacube supported!)"""
            raise NameError(message)
        else:
            self.param_file     = args.param
            self.outfile        = args.outfile
            self.level          = args.level
            self.norm           = args.norm
            self.rms            = args.rms

    def read_paramfile(self):
        """
        Function reading the parameter file provided by the command line
        argument, and defining class parameters.
        """
        param_file  = open(self.param_file, "r")
        params      = param_file.read()

        runlist_path = re.search(r"\nRUNLIST\s*=\s*'(\/.*?)'", params)  # Regex pattern to search for runlist path in parameter file.
        self.runlist_path = str(runlist_path.group(1))                  # Extracting path

        runlist_file = open(self.runlist_path, "r")         # Opening runlist file
        runlist = runlist_file.read()
        tod_in_list = re.findall(r"\/.*?\.\w+", runlist)    # Regex pattern to extract all L1 files to open from runlist.
        self.tod_in_list = tod_in_list                      

        patch_name = re.search(r"\s([a-zA-Z0-9]+)\s", runlist)  # Regex pattern to extract patch name of observations 
                                                                # (CO2, CO6 and CO7 supported)
        self.patch_name = str(patch_name.group(1))

        l1_in_path = re.search(r"\nTOD_OUT_DIR\s*=\s*'(\/.*?)'", params)    # Regex pattern to search for level1 file path.
        self.l1_in_path = str(l1_in_path.group(1))                          
    
        l2_in_path = re.search(r"\nLEVEL2_DIR\s*=\s*'(\/.*?)'", params)         # Regex pattern to search for level2 file path.
        self.l2_in_path = str(l2_in_path.group(1))  + f"/{self.patch_name}/"    
    
        l1_out_path = re.search(r"\nL1_OUT_DIR\s*=\s*'(\/.*?)'", params)    # Regex pattern to search for directory where to put the level1 maps.
        self.l1_out_path = str(l1_out_path.group(1))                          
        
        l2_out_path = re.search(r"\nL2_OUT_DIR\s*=\s*'(\/.*?)'", params)    # Regex pattern to search for directory where to put the level2 maps.
        self.l2_out_path = str(l2_out_path.group(1))                          
        
        map_in_path = re.search(r"\nMAP_DIR\s*=\s*'(\/.*?)'", params)    # Regex pattern to search for directory where to put the level2 maps.
        self.map_in_path = str(map_in_path.group(1))                          
        
        cube_path = re.search(r"\nDATACUBE\s*=\s*'(\/.*?\.\w+)'", params)   # Defining regex pattern to search for simulation cube file path.
        self.cube_filename = str(cube_path.group(1))                        # Extracting path

        cube_out_path = re.search(r"\nDATACUBE_OUT\s*=\s*'(\/.*?)'", params)   # Defining regex pattern to search for output simulation cube file path.
        self.cube_out_path = str(cube_out_path.group(1))                                # Extracting path

        obsIDs_list = re.findall(r"\s\d{6}\s", runlist)         # Regex pattern to find all obsIDs in runlist
        self.obsIDs_list = [int(i) for i in obsIDs_list]
        self.nobsIDs_list = len(self.obsIDs_list)               # Number of obsIDs in runlist

        patch_def_path = re.search(r"\nPATCH_DEFINITION_FILE\s*=\s*'(\/.*?)'", params)  # Regex pattern to search for patch definition file.
        self.patch_def_path = str(patch_def_path.group(1))

        patch_def_file = open(self.patch_def_path, "r")             # Opening patch definition file
        patch_def = patch_def_file.read()
        fieldcent   = re.search(rf"{self.patch_name}\s*([0-9.]+)\s*([0-9.]+)", patch_def) # Regex pattern to search for patch center
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
        print("Cube in:", self.cube_filename)
        print("Cube out:", self.cube_out_path)
        print("# obsID", len(self.tod_in_list))
        print("obsID file #1: ", self.tod_in_list[0])
        print("obsID #1: ", self.obsIDs_list[0])

    def run(self):
        """
        Function executing the projection from TOD to map.
        """

        self.read_paramfile()
        if self.level == "1":
            print("Loopig through runlist L1: ")
            for i in trange(len(self.tod_in_list)):
                self.infile    = self.l1_in_path + self.tod_in_list[i]
                self.obsID = self.obsIDs_list[i]
                self.readL1()
                self.make_mapL1()
                self.write_map()

            print("Through loop L1")

        elif self.level == "2":
            print("Loopig through runlist L2: ")
            for i in trange(len(self.tod_in_list)):
                self.obsID = self.obsIDs_list[i]
                l2_files = []
                for filename in os.listdir(self.l2_in_path):
                    if f"{self.obsID}" in filename: 
                        l2_files.append(filename)
                self.l2_files = l2_files
                self.make_mapL2()
                self.write_map()
            print("Through loop L2")
        else:
            print("Processing simulation cube: ")
            for i in trange(len(self.tod_in_list)):
                self.infile    = self.l1_in_path + self.tod_in_list[i]
                self.obsID = self.obsIDs_list[i]
                self.load_cube()                        

                for filename in os.listdir(self.map_in_path):
                    if f"{self.obsID}" in filename: 
                        self.map_in_name = self.map_in_path + filename
                self.make_map_cube()
                self.write_map()

            print("Cube processed!")


    def readL1(self):
        """
        Function opening and reading level1 file.
        """
        infile          = h5py.File(self.infile, "r")
        self.tod        = np.array(infile["spectrometer/tod"])[()].astype(dtype=np.float32, copy=False) 
        self.ra         = np.array(infile["spectrometer/pixel_pointing/pixel_ra"])[()]
        self.dec        = np.array(infile["spectrometer/pixel_pointing/pixel_dec"])[()]
        
        self.tod[:, 0, :, :] = self.tod[:, 0, ::-1, :]
        self.tod[:, 2, :, :] = self.tod[:, 2, ::-1, :]
        self.nfeeds, self.nsb, self.nfreq, self.nsamp = self.tod.shape

        infile.close()
    
    def readL2(self):
        """
        Function opening and reading level2 file.
        """
        #print("Reading L2 file:")
        infile      = h5py.File(self.infile, "r")
        self.tod    = np.array(infile["tod"])[()].astype(dtype=np.float32, copy=False) 
        pointing    = np.array(infile["point_cel"])[()]
        self.ra     = pointing[:, :, 0] 
        self.dec    = pointing[:, :, 1] 

        self.tod[:, 0, :, :] = self.tod[:, 0, ::-1, :]
        self.tod[:, 2, :, :] = self.tod[:, 2, ::-1, :]
        self.nfeeds, self.nsb, self.nfreq, self.nsamp = self.tod.shape
        infile.close()
    
    def load_cube(self):
        """
        Read the simulated datacube into memory.
        """
        cube = np.load(self.cube_filename)
        cubeshape = cube.shape

        cube /= self.norm    # Normalization of cube by input value

        cube = cube.reshape(cubeshape[0], cubeshape[1], 4, 1024)  # Flatten the x/y dims, and split the frequency (depth) dim in 4 sidebands.
        cube = cube.reshape(cubeshape[0], cubeshape[1], 4, 64, 16)
        cube = np.mean(cube, axis = 4)     # Averaging over 16 frequency channels
        cube = cube.transpose(2, 3, 0, 1)

        self.cube = cube
            
        infile          = h5py.File(self.infile, "r")
        self.freq         = np.array(infile["spectrometer/frequency"])[()]
        self.ra         = np.array(infile["spectrometer/pixel_pointing/pixel_ra"])[()]
        self.dec        = np.array(infile["spectrometer/pixel_pointing/pixel_dec"])[()]
        infile.close()
        
        self.nsb, self.nfreq    = self.freq.shape
        self.nfeeds, self.nsamp     = self.ra.shape
    
    def hist(self, idx, tod):
        """
        Function performing the binning of the TOD in pixel bins.

        Parameters:
        --------------------
        idx: array, dtype = ctypes.c_int
            Array of pixel number of flattened image, 
            i.e. the number of the pixel bin.
        tod: ndarray, dtype = ctypes.c_float
            Array with dimension (nsb, nfreq, nsamp) (i.e. per feed)
            which is binned up in the pixel bins given by
            idx array.
        --------------------
        Returns:
            map: ndarray, dtype = ctypes.c_float
                Array of shape (nsb, nfreq, nbin) corresponding to
                the flattened pixel image after bining.
            nhit: ndarray, dtype = ctypes.c_int
                Array of same shape as map corresponding to hits 
                in each pixel bin.
        """

        map = np.zeros((self.nsb, self.nfreq, self.nbin), dtype = ctypes.c_float)   # Array to be filled with TOD values at each pixel.
        nhit = np.zeros((self.nsb, self.nfreq, self.nbin), dtype = ctypes.c_int)    # Array to be filled with hits at each pixel.

        maputilslib = ctypes.cdll.LoadLibrary("histutils.so.1")                     # Load shared C utils library.
        
        float32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=3, flags="contiguous")       # 3D array 32-bit float pointer object.
        #float64_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=3, flags="contiguous")     # 3D array 64-bit float pointer object.
        int32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=3, flags="contiguous")           # 3D array 32-bit integer pointer object.
        int32_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, flags="contiguous")           # 1D array 32-bit integer pointer object.

        maputilslib.histogram.argtypes = [int32_array1, float32_array3, float32_array3, int32_array3,   # Specifying input types for C library function.
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
        maputilslib.histogram(idx, tod, map, nhit, self.nsb, self.nfreq, self.nsamp, self.nbin) # Filling map and nhit by call-by-pointer.
        
        return map, nhit

    def nhits(self, idx):
        """
        Function performing the binning of the TOD in pixel bins.

        Parameters:
        --------------------
        idx: array, dtype = ctypes.c_int
            Array of pixel number of flattened image, 
            i.e. the number of the pixel bin.
        --------------------
        Returns:
            nhit: ndarray, dtype = ctypes.c_int
                Array of shape (nsb, nfreq, nbin) corresponding to hits 
                in each pixel bin.
        """

        nhit = np.zeros((self.nsb, self.nfreq, self.nbin), dtype = ctypes.c_int)    # Array to be filled with hits at each pixel.

        maputilslib = ctypes.cdll.LoadLibrary("histutils.so.1")                     # Load shared C utils library.
        
        int32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=3, flags="contiguous")           # 3D array 32-bit integer pointer object.
        int32_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, flags="contiguous")           # 1D array 32-bit integer pointer object.
        maputilslib.nhits.argtypes = [int32_array1, int32_array3,   # Specifying input types for C library function.
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
        maputilslib.nhits(idx, nhit, self.nsb, self.nfreq, self.nsamp, self.nbin) # Filling map and nhit by call-by-pointer.
        
        return nhit


    def make_mapL1(self):
        """
        Function mapping the level1 TOD to the pixel regime. 
        """
        print("Making L1 map:") 
        histo = np.zeros((self.nsb, self.nfreq, self.nside, self.nside))  # Empty array of images to fill  
        allhits = np.zeros_like(histo)                                    # Empty array of hits to fill
        px_idx      = np.zeros_like(self.dec, dtype = ctypes.c_int)       # Empty array of pixel numbers
        looplen = 0                                                       # Loop iterator used for averaging over feeds
        
        for i in trange(self.nfeeds):  
            looplen += 1
            px_idx[i, :] = WCS.ang2pix([self.nside, self.nside], 
                                        [-self.dpix, self.dpix], 
                                        self.fieldcent, 
                                        self.dec[i, :], 
                                        self.ra[i, :])              # Finding pixel number corresponding to each Ra/Dec.
            map, nhit = self.hist(px_idx[i, :], self.tod[i, ...])   # Get image and nhit

            histo += map.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
            allhits += nhit.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
        
        histo      /= allhits       # Weighting by hits in given pixel
        histo       = np.nan_to_num(histo, nan = 0) # Changing NaNs to 0.
        
        histo = histo.reshape(self.nsb, int(self.nfreq / 16), 16, self.nside, self.nside)   # Averaging over 16 freq channels
        histo = np.nanmean(histo, axis = 2)

        allhits = allhits.reshape(self.nsb, int(self.nfreq / 16), 16, self.nside, self.nside)
        allhits = np.nansum(allhits, axis = 2)

        self.map    = histo / looplen   # Averaging over feeds
        self.nhit   = allhits
        #print("DOne with mapmaking:")

    def make_mapL2(self):
        """
        Function mapping the level2 TOD to the pixel regime. 
        """
        #print("Makign L2 map:")
        self.nsb, self.nfreq, self.nside = 4, 64, 120
        histo   = np.zeros((self.nsb, self.nfreq, self.nside, self.nside))  # Empty array to be filled with pixel image
        allhits = np.zeros_like(histo)                                      # Empty array to be filled with pixel hits
        looplen = 0                         # Loop iterator used for averaging over feeds.

        """Looping through all level2 files in given input directory"""
        for i in trange(len(self.l2_files)):
            self.infile   = self.l2_in_path + self.l2_files[i]
            self.readL2() 
            px_idx  = np.zeros_like(self.dec, dtype = ctypes.c_int) # Empty array to fill with pixel numbers
            for j in range(self.nfeeds):  
                looplen += 1
                px_idx[j, :] = WCS.ang2pix([self.nside, self.nside], 
                                            [-self.dpix, self.dpix], 
                                            self.fieldcent, 
                                            self.dec[j, :], 
                                            self.ra[j, :])              # Finding pixel number corresponding to each Ra/Dec.
                map, nhit = self.hist(px_idx[j, :], self.tod[j, ...])   # Get image and nhit  
                
                histo   += map.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
                allhits += nhit.reshape(self.nsb, self.nfreq, self.nside, self.nside)    
        
        histo      /= allhits   # Weighting by hits in given pixel
        
        histo       = np.nan_to_num(histo, nan = 0) # Changing NaNs to 0
        self.map    = histo #/ looplen               # Averaging over feeds 
        self.nhit   = allhits

    def make_map_cube(self):
        """
        Function mapping the simulated cube to the pixel regime of a map file. 
        """
        allhits = np.zeros((self.nsb, self.nfreq, self.nside, self.nside))                                    # Empty array of hits to fill
          
        px_idx      = np.zeros_like(self.dec, dtype = ctypes.c_int)       # Empty array of pixel numbers
        looplen = 0                                                       # Loop iterator used for averaging over feeds
        
        for i in trange(self.nfeeds):  
            looplen += 1
            px_idx[i, :] = WCS.ang2pix([self.nside, self.nside], 
                                        [-self.dpix, self.dpix], 
                                        self.fieldcent, 
                                        self.dec[i, :], 
                                        self.ra[i, :])              # Finding pixel number corresponding to each Ra/Dec.
            nhit = self.nhits(px_idx[i, :])                         # Get image and nhit

            allhits += nhit.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
        
        allhits = allhits.reshape(self.nsb, int(self.nfreq / 16), 16, self.nside, self.nside)
        allhits = np.nansum(allhits, axis = 2)

        self.cube   = np.where(allhits > 0, self.cube, 0)
        self.map    = self.cube    
        self.nhit   = allhits
        
    def copy_mapfile(self):
        """
        Function copying copying the template file to be filled
        with map and hits coadded over feeds.
        """
        shutil.copyfile(self.template_file, self.outfile)


    def write_map(self):
        if self.level == "1":
            self.map_out_path = self.l1_out_path
        elif self.level == "2":
            self.map_out_path = self.l2_out_path
        else:
            self.map_out_path = self.cube_out_path

        for template in os.listdir(self.template_path):
            if self.patch_name in template:
                self.template_file = self.template_path + template
        self.outfile = self.map_out_path + f"{self.patch_name}_{self.obsID}_{self.outfile}_map.h5"         
        self.copy_mapfile()

        if self.rms:     
            inmap   = h5py.File(self.map_in_name, "r")
            try:
                inrms     = np.array(inmap["rms_coadd"])[()]
            except KeyError:
                inrms     = np.array(inmap["rms_beam"])[()]
            rms     = inrms.copy()
        else:
            rms = np.ones_like(self.nhit)
            rms = np.where(self.nhit > 0, rms, 0)
            rms = rms.transpose(0, 1, 3, 2) 
            
        with h5py.File(self.outfile, "r+") as outfile:  # Write new sim-data to file.
            try:
                map_coadd   = outfile["map_coadd"] 
                nhit_coadd  = outfile["nhit_coadd"] 
                rms_coadd   = outfile["rms_coadd"] 
                
            except KeyError:
                map_coadd   = outfile["map_beam"] 
                nhit_coadd  = outfile["nhit_beam"] 
                rms_coadd   = outfile["rms_beam"] 
            
            map_coadd[...]  = self.map.transpose(0, 1, 3, 2)
            nhit_coadd[...] = self.nhit.transpose(0, 1, 3, 2)
            rms_coadd[...]  = rms 

        outfile.close()

if __name__ == "__main__":
    maker = MapMakerLight()
    #maker.read_paramfile()
    maker.run()
