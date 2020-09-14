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
        self.template_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/templates/"
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

        runlist_file = open(self.runlist_path, "r")         # Opening 
        runlist = runlist_file.read()
        tod_in_list = re.findall(r"\/.*?\.\w+", runlist)
        self.tod_in_list = tod_in_list

        patch_name = re.search(r"\s([a-zA-Z0-9]+)\s", runlist)
        self.patch_name = str(patch_name.group(1))

        l1_in_path = re.search(r"\nTOD_OUT_DIR\s*=\s*'(\/.*?)'", params)    # Defining regex pattern to search for level1 file path.
        self.l1_in_path = str(l1_in_path.group(1))                        # Extracting path
    
        l2_in_path = re.search(r"\nLEVEL2_DIR\s*=\s*'(\/.*?)'", params)    # Defining regex pattern to search for level1 file path.
        self.l2_in_path = str(l2_in_path.group(1))  + f"/{self.patch_name}/"  # Extracting path
    
        l1_out_path = re.search(r"\nL1_OUT_DIR\s*=\s*'(\/.*?)'", params)   # Defining regex pattern to search for directory where to put the level1 maps.
        self.l1_out_path = str(l1_out_path.group(1))                          # Extracting path
        
        l2_out_path = re.search(r"\nL2_OUT_DIR\s*=\s*'(\/.*?)'", params)   # Defining regex pattern to search for directory where to put the level2 maps.
        self.l2_out_path = str(l2_out_path.group(1))                          # Extracting path

        
        tod_in_list = re.findall(r"\/.*?\.\w+", runlist)
        obsIDs_list = re.findall(r"\s\d{6}\s", runlist)
        self.obsIDs_list = [int(i) for i in obsIDs_list]
        self.nobsIDs_list = len(self.obsIDs_list)
        


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
                self.outfile   = self.outfile
                        
                self.readL1()
                self.make_mapL1()
                self.write_map()

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
                self.write_map()
            print("Through loop L2")

    def readL1(self):
        t = time.time()
        infile          = h5py.File(self.infile, "r")
        self.tod        = np.array(infile["spectrometer/tod"])[()].astype(dtype=np.float32, copy=False) 
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
        map = np.zeros((self.nsb, self.nfreq, self.nbin), dtype = ctypes.c_float)
        nhit = np.zeros((self.nsb, self.nfreq, self.nbin), dtype = ctypes.c_int)

        maputilslib = ctypes.cdll.LoadLibrary("histutils.so.1")  # Load shared C utils library.
        float32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=3, flags="contiguous")   # 4D array 32-bit float pointer object.
        #float64_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=3, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=3, flags="contiguous")       # 4D array 32-bit integer pointer object.
        int32_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, flags="contiguous")       # 4D array 32-bit integer pointer object.

        maputilslib.histogram.argtypes = [int32_array1, float32_array3, float32_array3, int32_array3,        # Specifying input types for C library function.
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
        maputilslib.histogram(idx, tod, map, nhit, self.nsb, self.nfreq, self.nsamp, self.nbin)
        return map, nhit

    def make_mapL1(self):
        print("Making L1 map:") 
        histo = np.zeros((self.nsb, self.nfreq, self.nside, self.nside))    
        allhits = np.zeros_like(histo)    
        px_idx      = np.zeros_like(self.dec, dtype = ctypes.c_int)
        looplen = 0
                
        for i in trange(self.nfeeds):  
            looplen += 1
            px_idx[i, :] = WCS.ang2pix([self.nside, self.nside], 
                                        [-self.dpix, self.dpix], 
                                        self.fieldcent, 
                                        self.dec[i, :], 
                                        self.ra[i, :])
            map, nhit = self.hist(px_idx[i, :], self.tod[i, ...])

            histo += map.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
            allhits += nhit.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
        
        histo      /= allhits
        histo       = np.nan_to_num(histo, nan = 0)

        histo = histo.reshape(self.nsb, int(self.nfreq / 16), 16, self.nside, self.nside)
        histo = np.nanmean(histo, axis = 2)
        
        allhits = allhits.reshape(self.nsb, int(self.nfreq / 16), 16, self.nside, self.nside)
        allhits = np.nansum(allhits, axis = 2)
        
        self.map    = histo / looplen
        self.nhit   = allhits
        print("DOne with mapmaking:")

    def make_mapL2(self):
        print("Makign L2 map:")
        self.nsb, self.nfreq, self.nside = 4, 64, 120
        histo   = np.zeros((self.nsb, self.nfreq, self.nside, self.nside))
        allhits = np.zeros_like(histo)    
        looplen = 0

        for i in trange(len(self.l2_files)):
            self.infile   = self.l2_in_path + self.l2_files[i]
            self.readL2()
            px_idx  = np.zeros_like(self.dec, dtype = ctypes.c_int)
            for j in range(self.nfeeds):  
                looplen += 1
                px_idx[j, :] = WCS.ang2pix([self.nside, self.nside], 
                                            [-self.dpix, self.dpix], 
                                            self.fieldcent, 
                                            self.dec[j, :], 
                                            self.ra[j, :])
                map, nhit = self.hist(px_idx[j, :], self.tod[j, ...])
                
                histo   += map.reshape(self.nsb, self.nfreq, self.nside, self.nside)     
                allhits += nhit.reshape(self.nsb, self.nfreq, self.nside, self.nside)    

        histo      /= allhits
        histo       = np.nan_to_num(histo, nan = 0)
        self.map    = histo / looplen
        self.nhit   = allhits        

    def copy_mapfile(self):
        shutil.copyfile(self.template_file, self.outfile)


    def write_map(self):
        if self.level == 1:
            self.map_out_path = self.l1_out_path
        else:
            self.map_out_path = self.l2_out_path

        for template in os.listdir(self.template_path):
            if self.patch_name in template:
                self.template_file = self.template_path + template
        self.outfile = self.map_out_path + f"{self.patch_name}_{self.outfile}_map.h5"            
        self.copy_mapfile()

        with h5py.File(self.outfile, "r+") as outfile:  # Write new sim-data to file.
            try:
                map_coadd   = outfile["map_coadd"] 
                nhit_coadd  = outfile["nhit_coadd"] 
                rms_coadd   = outfile["rms_coadd"] 
                
            except KeyError:
                map_coadd   = outfile["map_beam"] 
                nhit_coadd  = outfile["nhit_beam"] 
                rms_coadd   = outfile["rms_beam"] 
                
            map     = outfile["map"] 
            nhit    = outfile["nhit"] 
            rms     = outfile["rms"] 

            map_coadd[...]  = self.map
            nhit_coadd[...] = self.nhit
            rms_coadd[...]  = np.ones_like(self.nhit)

            map[...]    = np.zeros((19, 4, 64, 120, 120))
            nhit[...]   = np.zeros((19, 4, 64, 120, 120))
            rms[...]    = np.zeros((19, 4, 64, 120, 120))
    
        outfile.close()

if __name__ == "__main__":
    maker = MapMakerLight()
    #maker.read_paramfile()
    maker.run()
