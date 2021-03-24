import numpy as np 
from scipy.sparse import csc_matrix, identity, diags
import sparse_dot_mkl
import scipy.sparse.linalg as linalg
from scipy import signal
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as multiproc
import time
import sys
import h5py
import WCS
import copy
import os
import re
import argparse
import warnings
import random

warnings.filterwarnings("ignore", category=RuntimeWarning) #ignore warnings caused by weights cut-off

class Destriper():
    def __init__(self, eps = 0, param_file = None, infile_path = None, outfile_path = None, obsID_map = False):
        self.infile_path    = infile_path
        self.outfile_path   = outfile_path
        self.obsID_map      = obsID_map 
        #self.N_baseline  = N_baseline
        #self.highcut     = highcut
        #self.baseline_time = 1 / highcut
        self.eps         = eps
        #self.scheme      = scheme
        self.param_file  = param_file 
        
        if param_file != None:
            self.read_paramfile()
        else:
            self.input()
            self.read_paramfile()
        
        if self.scheme not in ["destriper", "weighted", "avg", "baseline_only"]:
            print("Please provide one of the allowed mapmaker schemes: 'destriper', 'weighted', 'avg' or 'baseline_only'.")
            sys.exit()
        
        
        self.Nside = 120           # Number of pixels along RA/Dec
        self.Npix   = self.Nside ** 2   # Total number of pixels in the image
        
        self.dpix   = 2.0 / 60.0    # Pixel resolution in degrees (2' = 2/60 deg)
        self.obsID  = None

        self.cube_filename = "/mn/stornext/d16/cmbco/comap/protodir/cube_real.npy"        
        
        self.counter = 0
        self.counter2 = 0

    def input(self):
        """
        Function parsing the command line input.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--param", type = str, default = None,
                            help = """Full path and name to input parameter file.""")
        
        args = parser.parse_args()

        if args.param == None:
            message = """No input parameter file given, please provide an input parameter file!"""
            raise NameError(message)
        else:
            self.param_file     = args.param
        
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

        infile_path = re.search(r"\nLEVEL2_DIR\s*=\s*'(\/.*?)'", params)    # Regex pattern to search for level1 file path.
        self.infile_path = str(infile_path.group(1)) + "/" + self.patch_name + "/"                          
    
        outfile_path = re.search(r"\nMAP_DIR\s*=\s*'(\/.*?)'", params)    # Regex pattern to search for directory where to put the level2 maps.
        self.outfile_path = str(outfile_path.group(1))                          
        
        mapname = re.search(r"\nMAP_NAME\s*=\s*'([0-9A-Za-z\_]*)'", params)   # Defining regex pattern to search for output simulation cube file path.
        self.map_name = str(mapname.group(1))                                # Extracting path

        
        scanIDs = re.findall(r"\s\d{8}\s", runlist)         # Regex pattern to find all scanIDs in runlist
        self.scanIDs = [num.strip() for num in scanIDs]
        self.nscanIDs = len(self.scanIDs)                  # Number of scanIDs in runlist
        
        obsIDs = re.findall(r"\s\d{6}\s", runlist)         # Regex pattern to find all scanIDs in runlist
        self.obsIDs = [num.strip() for num in obsIDs]
        self.nobsIDs = len(self.obsIDs)                  # Number of scanIDs in runlist
        
        patch_def_path = re.search(r"\nPATCH_DEFINITION_FILE\s*=\s*'(\/.*?)'", params)  # Regex pattern to search for patch definition file.
        self.patch_def_path = str(patch_def_path.group(1))

        patch_def_file = open(self.patch_def_path, "r")             # Opening patch definition file
        patch_def = patch_def_file.read()
        fieldcent   = re.search(rf"{self.patch_name}\s*([0-9.]+)\s*([0-9.]+)", patch_def) # Regex pattern to search for patch center
        self.fieldcent = np.array([float(fieldcent.group(1)), float(fieldcent.group(2))])               

        highpass_nu = re.search(r"\nNUCUT_HIGH\s*=\s*([0-9.]+)", params)    # Regex pattern to search for directory where to put the level2 maps.
        self.highpass_nu = float(highpass_nu.group(1))                          
        
        basefreq = re.search(r"\nBASELINE_NU\s*=\s*([0-9.]+)", params)    # Regex pattern to search for directory where to put the level2 maps.
        self.basefreq = float(basefreq.group(1))                          
        self.baseline_time = 1 / self.basefreq
        
        scheme = re.search(r"\nSCHEME\s*=\s*'(\w+)'", params)  # Regex pattern to search for patch definition file.
        self.scheme = str(scheme.group(1))
        
        masking = re.search(r"\nUSE_MASK\s*=\s*([01])", params)  # Regex pattern to search for patch definition file.
        self.masking = bool(int(masking.group(1)))
        
        baseline_only = re.search(r"\nFIT_BASELINES\s*=\s*(.true.|.false.)", params)  # Regex pattern to search for patch definition file.
        self.baseline_only = baseline_only.group(1)
        if self.baseline_only == ".true.":
            self.baseline_only = True 
            self.scheme = "baseline_only"
        else:
            self.baseline_only = False 

        Nproc = re.search(r"\nN_FREQ_PROCESS\s*=\s*(\d+)", params)  # Regex pattern to search for number of frequency processes to run.

        self.Nproc = int(Nproc.group(1))
        
        perform_split = re.search(r"\nUSE_ACCEPT\s*=\s*(.true.|.false.)", params)  # Regex pattern to search for patch definition file.
        
        self.perform_split = perform_split.group(1)
        
        if self.perform_split == ".true.":
            self.perform_split = True 
            self.scheme = "baseline_only"
        else:
            self.perform_split = False 
        
        if self.perform_split: 
            # Read in filename for split data file
            accpt_data_path = re.search(r"\nACCEPT_DATA_FOLDER\s*=\s*'(\/.*?)'", params)   # Defining regex pattern to search for output simulation cube file path.
            accpt_data_path = str(accpt_data_path.group(1))                                # Extracting path

            accpt_id = re.search(r"\nACCEPT_DATA_ID_STRING\s*=\s*'([0-9A-Za-z\_]*)'", params)   # Defining regex pattern to search for output simulation cube file path.
            accpt_id = str(accpt_id.group(1))                                # Extracting path

            split_id = re.search(r"\nJK_DATA_STRING\s*=\s*'(\/.*?)'", params)   # Defining regex pattern to search for output simulation cube file path.
            if split_id == None:
                split_id = ""
            else:
                split_id = str(split_id.group(1))                                # Extracting path

            self.acceptfile_name = accpt_data_path + "jk_data_" +  accpt_id + split_id + "_" + self.patch_name + ".h5"

            # jk_list file from parameter file
            split_def = re.search(r"\nJK_DEF_FILE\s*=\s*'(\/.*?)'", params)   # Defining regex pattern to search for output simulation cube file path.
            self.split_def = str(split_def.group(1))                                # Extracting path


            idx_feed = np.arange(19, dtype = int)
            idx_sb   = np.arange(4,  dtype = int)
            idx_freq = np.arange(64, dtype = int)
 
            feeds, sidebands, frequencies = np.meshgrid(idx_feed, idx_sb, idx_freq, indexing = "ij")

            self.all_idx = np.array([feeds.flatten(), sidebands.flatten(), frequencies.flatten()])

        else:
            sidebands = np.arange(4, dtype = int)
            freqs     = np.arange(64, dtype = int)

            sidebands, freqs = np.meshgrid(sidebands, freqs, indexing = "ij")

            self.all_idx = np.array([sidebands.flatten(), freqs.flatten()])

        runlist_file.close()    
        param_file.close()


        print("Patch def:", self.patch_def_path)
        print("Patch", self.patch_name)
        print("Field center", self.fieldcent)
        print("Runlist:", self.runlist_path)
        print("Infile path:", self.infile_path)
        print("Outfile path:", self.outfile_path)
        print("scan IDs:", self.scanIDs)
        print("obs IDs:", self.obsIDs)
        print("Number of scans:", self.nscanIDs)
        print("Number of obs:", self.nobsIDs)
        print("Map output name:", self.map_name)
        print("Mapmaker scheme:", self.scheme)
        print("Fit baseline only:", self.baseline_only)

        print("Baseline freq:", self.basefreq)
        print("Highpass cut:", self.highpass_nu)
        print("Use mask:", self.masking)
        print("Number of frequency loop processes:", self.Nproc)


        if self.perform_split:
            print("Perform split:", self.perform_split)
            print("Accept data_folder: ", accpt_data_path)
            print("Accept ID: ", accpt_id)
            print("Split ID: ", split_id)
            print("Split data file:", self.acceptfile_name)
            print("Split def file:", self.split_def)
            self.read_split_def()
            self.read_split_data()
            self.read_feeds_alive()
            
            
    def run(self, feed = 1, sb = 1, freq = 1, freq_idx = None):
        t0 = time.time()
        if freq_idx == None:
            self.feed = feed
            self.sb   = sb 
            self.freq = freq
        else: 
            self.freq_idx = freq_idx
            if self.perform_split:
                feed, sb, freq = self.all_idx[:, freq_idx]
                self.feed, self.sb, self.freq = feed, sb, freq
            else:    
                sb, freq = self.all_idx[:, freq_idx]
                self.sb, self.freq = sb, freq

        if self.perform_split:
            self.get_split_data()
            #print("Get P and F:"); t0 = time.time()
            self.initialize_P_and_F()
            #print("Get P and F time:", time.time() - t0, "sec")
        else:
            self.tod    = self.tod_buffer.reshape(self.Nsamp, 4, 64)[:, sb, freq] 
            self.sigma0 = self.sigma0_buffer.reshape(self.Nscans, 4, 64)[:, sb, freq]
            self.mask   = self.mask_buffer.reshape(self.Nscans, 4, 64)[:, sb, freq] 

        if self.masking:
            #print("Get pointing:");t0 = time.time()
            self.get_P()
            #print("Get pointing time:", time.time() - t0, "sec")

        #print("Get C_n_inv:")
        t0 = time.time()
        self.sigma0_inv = 1 / self.sigma0
        self.sigma0_inv[self.mask == 0] = 0
        
        self.get_Cn_inv()
        #print("Get C_n_inv time:", time.time() - t0, "sec")

        #print("Get PCP_inv:")
        t0 = time.time()
        self.get_PCP_inv()
        #print("Get PCP_inv time:", time.time() - t0, "sec")
        
    def read_feeds_alive(self):
        feeds_alive = {}
        
        for filename in os.listdir(self.infile_path):
            if np.any([(name in filename and len(filename) < 17) for name in self.scanIDs]):
                infile = h5py.File(self.infile_path + filename, "r")
                feeds  = infile["pixels"][()]
                feeds_alive[filename] = feeds - 1       # Translating base 1 to base 0 indexing
                infile.close()

        self.feeds_alive = feeds_alive

    def read_split_def(self):
        split_names = []
        n_coadd = n_feedmap = n_ctrl = n_test = n_split = 0

        with open(self.split_def, "r") as split_def_file:
            n_split = int(split_def_file.readline().split()[0]) - 1
            split_def_file.readline()
            for line in split_def_file:
                line_element = line.split()
                split_names.append(line_element[0])
                number   = int(line_element[1])
                
                if number == 0:
                    n_coadd += 1
                elif number == 2:
                    n_test += 1
                elif number == 3:
                    n_ctrl += 1
                else:
                    n_feedmap += 1

        self.n_coadd     = n_coadd
        self.n_feedmap   = n_feedmap
        self.n_test      = n_test
        self.n_ctrl      = n_ctrl
        self.n_split     = n_split
        self.split_names = split_names  

        split_def_file.close()

    def read_split_data(self):
        
        with h5py.File(self.acceptfile_name, "r") as split_file:
            split_list  = split_file["jk_list"][()]
            split_scans = split_file["scan_list"][()] 
        split_file.close()

        self.split_scans = split_scans
        self.all_scans, nfeed, nsb = split_list.shape
        self.splits = np.zeros((self.n_split, self.all_scans, nfeed, nsb))
       
        for i in range(self.n_split):
            x, y, z = np.where(split_list > 2 ** i)
            self.splits[i, x, y, z] = 1
            split_list[x, y, z] -= 2 ** i

        self.n_split -= self.n_test + self.n_ctrl

        self.Nbatch = 2 ** self.n_split
        
        tod_lens  = []
        names     = []
        Nscans    = 0
        t = time.time()
        file_list = os.listdir(self.infile_path)
            
        for scan in self.scanIDs:
            for filename in file_list:
                if scan in filename and len(filename) < 17:
                    #print(scan, filename)
                    Nscans += 1
                    infile = h5py.File(self.infile_path + filename, "r")
                    tod_shape  = infile["tod"].shape
                    Nfeed, Nsb, Nfreq, Nsamp = tod_shape
                    Nfeed  -= 1
                    tod_lens.append(Nsamp)
                    names.append(filename)
                    infile.close()
        names = np.array(names)
        tod_lens = np.array(tod_lens)
        
        scan_idx = np.arange(Nscans, dtype = int)
        np.random.shuffle(scan_idx)
        
        self.names = names[scan_idx]
        self.tod_lens = tod_lens[scan_idx]
        self.splits = self.splits[:, scan_idx, :, :]
        self.split_scans = self.split_scans[scan_idx]
        self.Nscans = Nscans

        exp_of_two = 2 ** np.arange(self.n_split)
        self.batch_def = np.sum(self.splits * exp_of_two[:, np.newaxis, np.newaxis, np.newaxis], axis = 0)

    def get_split_data(self):
        currentNames, feed, sb, freq = self.currentNames, self.feed, self.sb, self.freq
        N_in_batch = len(currentNames)
       
        tod_lens  = []

        for i in range(N_in_batch):
            filename = currentNames[i]

            infile = h5py.File(self.infile_path + filename, "r")
            tod_shape  = infile["tod"].shape
            Nfeed, Nsb, Nfreq, Nsamp = tod_shape
            tod_lens.append(Nsamp)
            infile.close()

        self.Nfeed = Nfeed
        self.Nsb   = Nsb
        self.Nfreq = Nfreq


        tod_lens = np.array(tod_lens)
        self.tod_lens = tod_lens
        tod_cumlen = np.zeros(N_in_batch + 1).astype(int)
        tod_cumlen[1:] = np.cumsum(tod_lens).astype(int)
        Nsamp_tot = np.sum(tod_lens)
    
        # Computing the length of each baseline
        infile = h5py.File(self.infile_path + currentNames[0], "r")
        tod_time   = infile["time"][:2] * 3600 * 24
        infile.close()
       
        dt    = tod_time[1] - tod_time[0]

        Nperbaseline = int(round(self.baseline_time / dt))

        Nbaseline = np.floor(tod_lens / Nperbaseline).astype(int)

        excess = tod_lens - Nperbaseline * Nbaseline

        Nbaseline_tot = np.sum(Nbaseline) + np.sum(excess != 0)
        
        Nperbaselines = [0]
        scan_per_baseline = []
        tu = time.time()
        for i in range(N_in_batch):
            Nbaseline_in_scan = 0
            for j in range(Nbaseline[i]):
                Nperbaselines.append(Nperbaseline)
                scan_per_baseline.append(currentNames[i])
                Nbaseline_in_scan += 1
            if excess[i] > 0:
                Nperbaselines.append(excess[i])
                scan_per_baseline.append(currentNames[i])
                Nbaseline_in_scan += 1

        self.ra  = np.zeros(Nsamp_tot, dtype = np.float32)
        self.dec = np.zeros(Nsamp_tot, dtype = np.float32)
        self.tod = np.zeros(Nsamp_tot, dtype = np.float32)
        
        self.sigma0 = np.zeros(N_in_batch, dtype = np.float32)
        self.mask   = np.zeros(N_in_batch, dtype = np.uint8)
        
        self.start_stop = np.zeros((2, N_in_batch), dtype = int)
        
        for i in range(N_in_batch):
            #ti = time.time()
            name   = currentNames[i]
            
            alive = self.feeds_alive[name]
            feed_idx  = np.where(alive == feed)[0]

            if feed_idx.size == 1:
                #if i == 0:
                #    print("Loader:", self.freq_idx, feed, sb, freq, N_in_batch, "|", alive[feed_idx][0], feed, feed_idx[0])
                start = tod_cumlen[i]
                end   = tod_cumlen[(i + 1)]
                self.start_stop[0, i] = start
                self.start_stop[1, i] = end
                infile = h5py.File(self.infile_path + name, "r")

                self.mask[i]        = infile["freqmask"][feed_idx, sb, freq]
                self.sigma0[i]      = infile["sigma0"][feed_idx, sb, freq]
                self.tod[start:end] = infile["tod"][feed_idx, sb, freq, :] #[()]#.astype(dtype=np.float32, copy=False) 
                self.ra[start:end]  = infile["point_cel"][feed_idx, :, 0] 
                self.dec[start:end] = infile["point_cel"][feed_idx, :, 1] 
                infile.close()
            else:
                #print("Passing this feed idx", feed_idx, feed)
                continue
            #print("Loading scan: ", i, " / ", N_in_batch, ", ", name, ", Time:", time.time() - ti)
        
        self.dt = dt
        self.Nbaseline    = Nbaseline_tot
        self.Nperbaselines = np.array(Nperbaselines)
        self.Nsamp        = Nsamp_tot
        self.Nscans       = N_in_batch
        self.tod_cumlen   = tod_cumlen
        self.scan_per_baseline = np.array(scan_per_baseline)

    def get_data(self):
        tod_lens  = []
        names     = []
        Nscans    = 0
        t = time.time()
        files = os.listdir(self.infile_path)
        random.shuffle(files)
        if self.obsID_map:
            print("obsID ", self.obsID)
            for filename in os.listdir(self.infile_path):
                if self.obsID in filename and len(filename):
                    if np.any([(name in filename) for name in self.scanIDs]):
                        Nscans += 1
                        infile = h5py.File(self.infile_path + filename, "r")
                        tod_shape  = infile["tod"].shape
                        Nfeed, Nsb, Nfreq, Nsamp = tod_shape
                        Nfeed  -= 1
                        for i in range(Nfeed):
                            tod_lens.append(Nsamp)
                        names.append(filename)
                        infile.close()
                    else:
                        print("No scan detected for this obsID!")
        else:
            for filename in files:
                if np.any([(name in filename and len(filename) < 17) for name in self.scanIDs]):
                    Nscans += 1
                    infile = h5py.File(self.infile_path + filename, "r")
                    tod_shape  = infile["tod"].shape
                    Nfeed, Nsb, Nfreq, Nsamp = tod_shape
                    Nfeed  -= 1
                    for i in range(Nfeed):
                        tod_lens.append(Nsamp)
                    names.append(filename)
                    infile.close()
        self.Nfeed = Nfeed
        self.Nsb = Nsb
        self.Nfreq = Nfreq
        self.names = names

        infile = h5py.File(self.infile_path + names[1], "r")
        freq       = infile["nu"][0, ...]
        freq[0, :] = freq[0, ::-1]
        freq[2, :] = freq[2, ::-1]   
        self.Freq  = freq
        infile.close()
        
        print("Number of scans:", Nscans)
        
        tod_lens = np.array(tod_lens)
        self.tod_lens = tod_lens
        tod_cumlen = np.zeros(Nscans * Nfeed + 1).astype(int)
        tod_cumlen[1:] = np.cumsum(tod_lens).astype(int)
        Nsamp_tot = np.sum(tod_lens)
    
        self.tod_buffer = np.zeros((Nsamp_tot, Nsb, Nfreq), dtype = np.float32)
       
        #time_buffer = np.zeros(Nsamp_tot)
        ra_buffer = np.zeros(Nsamp_tot, dtype = np.float32)
        dec_buffer = np.zeros(Nsamp_tot, dtype = np.float32)
        self.sigma0_buffer = np.zeros((Nscans, Nfeed, Nsb, Nfreq), dtype = np.float32)
        self.mask_buffer = np.zeros((Nscans, Nfeed, Nsb, Nfreq), dtype = np.uint8)
        
        Nbaseline_tot = 0
        Nperbaselines = [0]
        Nperscan      = [0]

        self.start_stop = np.zeros((2, Nscans), dtype = int)
        
        for i in range(Nscans):
            infile = h5py.File(self.infile_path + names[i], "r")
            print("Loading scan: ", i, ", ", names[i])
            freqmask          = infile["freqmask"][:-1, ...]

            freqmask[:, 0, :] = freqmask[:, 0, ::-1] #[()]#.astype(dtype=np.float32, copy=False) 
            freqmask[:, 2, :] = freqmask[:, 2, ::-1] #[()]#.astype(dtype=np.float32, copy=False) 
            
            tod                = infile["tod"][:-1, ...] #[()]#.astype(dtype=np.float32, copy=False) 
            
            tod[:, 0, :, :]    = tod[:, 0, ::-1, :] #[()]#.astype(dtype=np.float32, copy=False) 
            tod[:, 2, :, :]    = tod[:, 2, ::-1, :] #[()]#.astype(dtype=np.float32, copy=False) 

            if tod.dtype != np.float32:
                raise ValueError("The input TOD should be of dtype float32!")

            tod_time   = infile["time"][()] * 3600 * 24
            
            sigma0 = infile["sigma0"][:-1, ...]
            
            sigma0[:, 0, :]    = sigma0[:, 0, ::-1] #[()]#.astype(dtype=np.float32, copy=False) 
            sigma0[:, 2, :]    = sigma0[:, 2, ::-1] #[()]#.astype(dtype=np.float32, copy=False) 

            ra        = infile["point_cel"][:-1, :, 0] 
            dec       = infile["point_cel"][:-1, :, 1] 

            Nsamp = tod.shape[-1] 
            Nperscan.append(Nsamp)
            dt    = tod_time[1] - tod_time[0]
                        
            Nperbaseline = int(round(self.baseline_time / dt))
            #N_baseline = int(round((tod_time[-1] - tod_time[0]) / self.baseline_time))
            Nbaseline = int(np.floor(Nsamp / Nperbaseline))
            #print(N_perbaseline, dt * N_perbaseline, int(round(self.baseline_time / dt)))
            excess = Nsamp - Nperbaseline * Nbaseline
            for j in range(Nfeed):
                Nbaseline_tot += Nbaseline
                
                for k in range(Nbaseline):
                    Nperbaselines.append(Nperbaseline)
            
                if excess > 0:
                    Nbaseline_tot += 1
                    Nperbaselines.append(excess)
                
                self.sigma0_buffer[i, j, ...] = sigma0[j, ...]
                self.mask_buffer[i, j, ...]   = freqmask[j, ...]
            
            start = tod_cumlen[i * Nfeed]
            end   = tod_cumlen[(i + 1) * Nfeed]
            self.start_stop[0, i] = start
            self.start_stop[1, i] = end

            tod = tod.transpose(0, 3, 1, 2)
            tod = tod.reshape(tod.shape[0] * tod.shape[1], Nsb, Nfreq)
            self.tod_buffer[start:end, ...]  = tod

            ra_buffer[start:end]   = ra.flatten()
            dec_buffer[start:end]  = dec.flatten()
            
            infile.close()

        self.dt = dt
        
        self.sigma0_buffer       = self.sigma0_buffer.reshape(Nscans * Nfeed, Nsb, Nfreq)
        self.mask_buffer         = self.mask_buffer.reshape(Nscans * Nfeed, Nsb, Nfreq)
        
        self.ra           = ra_buffer        #np.trim_zeros(ra_buffer)
        self.dec          = dec_buffer       #np.trim_zeros(dec_buffer)
        self.Nbaseline    = Nbaseline_tot
        self.Nperbaselines = np.array(Nperbaselines)
        self.Nsamp        = Nsamp_tot
        self.Nscans       = Nscans * Nfeed
        self.tod_cumlen   = tod_cumlen
        
        self.tod_buffer          = self.tod_buffer.reshape(   self.Nsamp,  Nsb * Nfreq) 
        self.sigma0_buffer       = self.sigma0_buffer.reshape(self.Nscans, Nsb * Nfreq)
        self.mask_buffer         = self.mask_buffer.reshape(  self.Nscans, Nsb * Nfreq)

    def initialize_P_and_F(self):
        #print("Get pixel index:")
        t0 = time.time()
        self.get_px_index()
        #print("Get pixel time:", time.time() - t0, "sec")

        if self.masking == False:
            print("Not masking:")
            #print("Get pointing matrix:")
            t0 = time.time()
            self.mask = np.ones(self.Nscans)
            self.get_P()
            #print("Get pointing matrix time:", time.time() - t0, "sec")

        #print("Get F:");t0 = time.time()
        t0 = time.time()
        self.get_F()
        #print("Get F time:", time.time() - t0, "sec")
        
    def get_xy(self):
        Nside, dpix, fieldcent, ra, dec = self.Nside, self.dpix, self.fieldcent, self.ra, self.dec
        
        #self.outfile = self.map_out_path + f"{self.patch_name}_{self.out_name}"         
        #print(self.outfile)
        
        x = np.zeros(Nside)
        y = np.zeros(Nside)
        dx = dpix / np.cos(np.radians(fieldcent[1]))
        dy = dpix 
        
        if Nside % 2 == 0:
            x_min = fieldcent[0] - dx * Nside / 2 
            y_min = fieldcent[1] - dy * Nside / 2  
            
        else: 
            x_min = fieldcent[0] - dx * Nside / 2 - dx / 2
            y_min = fieldcent[1] - dy * Nside / 2  - dy / 2
            
        x[0] = x_min + dx / 2
        y[0] = y_min + dy / 2
        
        for i in range(1, Nside):
            x[i] = x[i - 1] + dx
            y[i] = y[i - 1] + dy
        
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        
    def get_px_index(self):
        self.get_xy()
        Nside, dpix, fieldcent, ra, dec, dx, dy = self.Nside, self.dpix, self.fieldcent, self.ra, self.dec, self.dx, self.dy
        
        #self.px = WCS.ang2pix([Nside, Nside], [-dpix, dpix], fieldcent, dec, ra)
        ra_min, dec_min = self.x[0], self.y[0]
        #self.px = np.zeros(self.Nsamp)
        #for i in range(self.Nsamp):
        self.px = np.round((ra - ra_min) / dx) * Nside + np.round((dec - dec_min) / dy)
        self.px = self.px.astype(int)
        
    def get_P(self):
        Nsamp, Npix, Nscans, cumlen, mask = self.Nsamp, self.Npix, self.Nscans, self.tod_cumlen, self.mask

        tt = time.time()

        hits = np.ones(Nsamp)
        if self.masking:
            for i in range(Nscans):
                start = cumlen[i]
                end = cumlen[i + 1]
                hits[start:end] = mask[i]
        rows = np.arange(0, Nsamp, 1)
        cols = self.px    
        if np.any(cols < 0):
            cols[cols < 0] = 0    
        
        #print("Filling P1:", time.time() - tt, "sec");tt = time.time()
        
        self.P = csc_matrix((hits, (rows, cols)), shape = (Nsamp, Npix), dtype = np.uint8)
        #print("Filling P2:", time.time() - tt, "sec");tt = time.time()
        self.PT = csc_matrix(self.P.T, dtype = np.uint8)
        #print("Filling PT:", time.time() - tt, "sec")
        
    def get_F(self):
        Nsamp, Nbaseline, Nperbaselines = self.Nsamp, self.Nbaseline, self.Nperbaselines
        #tf = time.time()

        Nperbaselines_cum = np.zeros(Nbaseline + 1)
        Nperbaselines_cum = np.cumsum(Nperbaselines)
        
        ones = np.ones(Nsamp)
        rows = np.arange(0, Nsamp, 1)
        cols = np.zeros(Nsamp)
        #print("Precomputation of F", time.time() - tf, "sec"); tf = time.time()
        for i in range(Nbaseline):
            start = Nperbaselines_cum[i]
            end = Nperbaselines_cum[i + 1]
            cols[start:end] = np.tile(i, Nperbaselines[i + 1])
        #print("Filling F1:", time.time() - tf, "sec");tf = time.time()
        
        self.F = csc_matrix((ones, (rows, cols)), shape = (Nsamp, Nbaseline), dtype = np.uint8)
        #print("Filling F2:", time.time() - tf, "sec");tf = time.time()
        self.FT = csc_matrix(self.F.T, dtype = np.uint8)
        #print("Filling FT:", time.time() - tf, "sec")
    
    def get_Cn_inv(self):
        Nsamp, Nscans, cumlen = self.Nsamp, self.Nscans, self.tod_cumlen
        C_n_inv = np.zeros(Nsamp)

        for i in range(Nscans):
            start = cumlen[i]
            end = cumlen[i + 1]
            C_n_inv[start:end] = self.sigma0_inv[i] ** 2

        self.C_n_inv = diags(C_n_inv)
    
    def get_PCP_inv(self):
        #P, PT, C_n_inv, eps = self.P, self.PT, self.C_n_inv, self.eps
        
        PCP_inv = self.PT.dot(self.C_n_inv)        
        PCP_inv = PCP_inv.dot(self.P) + diags(self.eps * np.ones(self.Npix))
           
        self.PCP_inv = diags(1 / PCP_inv.diagonal(), format = "csc", dtype = np.float32)
   
    def get_FT_C_P_PCP(self):
        #FT, C_n_inv, P, PCP_inv = self.FT, self.C_n_inv, self.P, self.PCP_inv    
        
        FT_C = self.FT.dot(self.C_n_inv)        
        FT_C_P = FT_C.dot(self.P)
        
        self.FT_C_P_PCP = FT_C_P.dot(self.PCP_inv)
        
    def get_PT_C(self):
        #PT, C_n_inv = self.PT, self.C_n_inv
        
        self.PT_C = self.PT.dot(self.C_n_inv)
        
    def get_FT_C(self):
        #FT, C_n_inv = self.FT, self.C_n_inv

        self.FT_C = self.FT.dot(self.C_n_inv)        
    
    def Ax(self, a):
        #FT_C_F, FT_C_P_PCP, PT_C_F = self.FT_C_F, self.FT_C_P_PCP, self.PT_C_F
      
        temp0 = self.FT_C_F.dot(a)
        temp1 = self.PT_C_F.dot(a)
        temp2 = self.FT_C_P_PCP.dot(temp1)
        
        #print(a)
        #print("CG counter: ", self.counter)
        self.counter += 1
        
        if np.any(np.isnan(a)) or np.any(np.isinf(a)):
            print("NaN or Inf in template vector a!", " All NaN or Inf:", np.all(np.isnan(a)), np.all(np.isinf(a)), "Freq:", self.sb, self.freq, np.any(np.isnan(temp0)), np.any(np.isnan(temp1)), np.any(np.isnan(temp2)))
            sys.exit()

        return temp0 - temp2
        
    def b(self, x):        
        #FT_C, FT_C_P_PCP, PT_C = self.FT_C, self.FT_C_P_PCP, self.PT_C

        temp0 = self.FT_C.dot(x)
        temp1 = self.PT_C.dot(x)
        temp2 = self.FT_C_P_PCP.dot(temp1)
        
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("NaN or Inf in x!" " All NaN or Inf:", np.all(np.isnan(x)), np.all(np.isinf(x)), "Freq:", self.sb, self.freq, np.any(np.isnan(temp0)), np.any(np.isnan(temp1)), np.any(np.isnan(temp2)))
            sys.exit()
        return temp0 - temp2

    def get_baselines(self):
        self.get_FT_C()
        self.get_FT_C_P_PCP()
        self.get_PT_C()
        
        self.FT_C_F = self.FT_C.dot(self.F)
        self.PT_C_F = self.PT_C.dot(self.F)

        Ax = linalg.LinearOperator((self.Nbaseline, self.Nbaseline) , matvec = self.Ax)
        b  = self.b(self.tod)
        
        #print("Initializing CG:")
        self.a, info = linalg.cg(Ax, b)
        #print("CG final count: ", self.counter)
        
        self.counter = 0
        self.counter2 = 0

    
    def get_destriped_map(self):
        #P, PT, C_n_inv, F, tod, Nsamp, eps, PCP_inv = self.P, self.PT, self.C_n_inv, self.F, self.tod, self.Nsamp, self.eps, self.PCP_inv

        self.get_baselines()

        m = self.PCP_inv.dot(self.PT).dot(self.C_n_inv).dot(self.tod - self.F.dot(self.a))
        self.m = m.reshape(self.Nside, self.Nside)

    def get_bin_averaged_map(self):
        #P, PT, tod, eps = self.P, self.PT, self.tod, self.eps
        
        cut = self.highpass_nu
        if self.highpass_nu > 0:
            highpass = True        
        else:
            highpass = False
            
        if highpass:
            self.tod = np.fft.rfft(self.tod)
            freqs   = np.fft.rfftfreq(self.Nsamp, self.dt)
            cut_idx = np.argmin(np.abs(freqs - cut))
            self.tod[0:cut_idx] = 0
            self.tod     = np.fft.irfft(self.tod[1:])
            #sos = signal.butter(10, cut, "highpass", output = "sos")
            #tod = signal.sosfilt(sos, tod)
            
        A = self.PT.dot(self.P) + diags(self.eps * np.ones(self.Npix))
        b = self.PT.dot(self.tod)
        
        #A = sparse_dot_mkl.dot_product_mkl(PT, P) + diags(eps * np.ones(self.Npix))
        #b = sparse_dot_mkl.dot_product_mkl(PT, tod)
        self.m = linalg.spsolve(A, b).reshape(self.Nside, self.Nside)
    
    def get_noise_weighted_map(self):
        #self.get_P()
        #self.get_Cn_inv()
        
        cut = self.highpass_nu
        if self.highpass_nu> 0:
            highpass = True
        else:
            highpass = False
        
        #P, PT, C_n_inv, tod, Nsamp, eps, PCP_inv = self.P, self.PT, self.C_n_inv, self.tod, self.Nsamp, self.eps, self.PCP_inv 
        
        if highpass:
            print("Highpass filtering TOD")
            self.tod = np.fft.rfft(self.tod)
            freqs   = np.fft.rfftfreq(self.Nsamp, self.dt)
            cut_idx = np.argmin(np.abs(freqs - cut))
            self.tod[0:cut_idx] = 0
            self.tod     = np.fft.irfft(self.tod, n = self.Nsamp)
            #sos = signal.butter(10, cut, "highpass", output = "sos")
            #tod = signal.sosfilt(sos, tod)
            
        #PCP_inv = P.transpose().dot(C_n_inv)        
        #PCP_inv = PCP_inv.dot(P) + diags(eps * np.ones(self.Npix))
        #PCP_inv = diags(1 / PCP_inv.diagonal())
        
        self.m = self.PCP_inv.dot(self.PT).dot(self.C_n_inv).dot(self.tod).reshape(self.Nside, self.Nside)
    
    def make_map(self):
        if self.scheme == "destriper":
            self.get_destriped_map()
        elif self.scheme == "weighted":
            self.get_noise_weighted_map()
        else:
            self.get_bin_averaged_map()
    
    def make_baseline_only(self):
        self.get_baselines()
        #self.baseline_tod = self.F.dot(self.a) 
        #self.baseline_tod = self.tod.copy() 

    def get_hits(self):
        #self.get_P()
        #P, PT, Nside, Nsamp = self.P, self.PT, self.Nside, self.Nsamp
        #ones = np.ones(Nsamp)
        self.hits = self.PT.dot(self.P).diagonal().reshape(self.Nside, self.Nside)
        
    def get_rms(self):
        #self.get_P()
        #self.get_Cn_inv()
        #P, C_n_inv, Nsamp, Nside, eps, PCP_inv = self.P, self.C_n_inv, self.Nsamp, self.Nside, self.eps, self.PCP_inv
        
        #PCP_inv = P.transpose().dot(C_n_inv)        
        #PCP_inv = PCP_inv.dot(P) + diags(eps * np.ones(self.Npix))
        #PCP_inv = diags(1 / PCP_inv.diagonal())
        self.rms = np.sqrt(self.PCP_inv.diagonal().reshape(self.Nside, self.Nside))
        
    def load_cube(self):
        """
        Read the simulated datacube into memory.
        """
        cube = np.load(self.cube_filename)
        cubeshape = cube.shape

        cube *= 1e-6    # Normalization of cube by input value
        cube *= 1000     # Normalization of cube by input value
        print("MAX CUBE:", np.nanmax(cube))
        cube = cube.reshape(cubeshape[0], cubeshape[1], 4, 1024)  # Flatten the x/y dims, and split the frequency (depth) dim in 4 sidebands.
        cube = cube.reshape(cubeshape[0], cubeshape[1], 4, 64, 16)
        cube = np.mean(cube, axis = 4)     # Averaging over 16 frequency channels
        print(cube.shape)
        cube = cube.transpose(2, 3, 0, 1)
        print(cube.shape)

        self.cube = cube[self.sb, self.freq, :, :]
        
    def write_map(self, full_map, full_hits, full_rms):
        Nside, dpix, fieldcent, ra, dec, freq = self.Nside, self.dpix, self.fieldcent, self.ra, self.dec, self.Freq
        
        outfile = self.outfile_path + self.patch_name + "_" + self.map_name + ".h5"
        #self.outfile = self.map_out_path + f"{self.patch_name}_{self.out_name}"         
        #print(self.outfile)
        
        x = np.zeros(Nside)
        y = np.zeros(Nside)
        dx = dpix / np.cos(np.radians(fieldcent[1]))
        dy = dpix 
        
        if Nside % 2 == 0:
            x_min = fieldcent[0] - dx * Nside / 2 
            y_min = fieldcent[1] - dy * Nside / 2  
            
        else: 
            x_min = fieldcent[0] - dx * Nside / 2 - dx / 2
            y_min = fieldcent[1] - dy * Nside / 2  - dy / 2
            
        x[0] = x_min + dx / 2
        y[0] = y_min + dy / 2
        
        for i in range(1, Nside):
            x[i] = x[i - 1] + dx
            y[i] = y[i - 1] + dy
        
        full_map = np.where(np.isnan(full_map) == False, full_map, 0)
        full_hits = np.where(np.isnan(full_hits) == False, full_hits, 0)
        full_rms = np.where(np.isnan(full_rms) == False, full_rms, 0)
        
        with h5py.File(outfile, "w") as outfile:  # Write new sim-data to file.
            outfile.create_dataset("map_coadd",    data = full_map, dtype = "float32")
            outfile.create_dataset("nhit_coadd",   data = full_hits, dtype = "int32")
            outfile.create_dataset("rms_coadd",    data = full_rms, dtype = "float32")
            outfile.create_dataset("x",            data = x)
            outfile.create_dataset("y",            data = y)
            outfile.create_dataset("n_x",          data = Nside)
            outfile.create_dataset("n_y",          data = Nside)
            outfile.create_dataset("patch_center", data = fieldcent)
            outfile.create_dataset("freq",         data = freq)
            
        outfile.close()

    def save_baseline_tod(self, baseline_buffer):
        print("Saving baselines:")
        tod_lens = self.tod_lens
        tod_lens = tod_lens[::self.Nfeed]

        #outfile_path = self.infile_path + "baselines/"
        outfile_path = self.infile_path + "all_in_one/"
        #outfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/sim/highpass/002Hz/default/XL_dataset/co6/baselines/"
        #outfile_path = self.infile_path + "null_test/"
        
        print("Saveing baselines to:", outfile_path)
        if not os.path.exists(outfile_path):
            os.mkdir(outfile_path)
        
        for i in range(len(self.names)):
            start, stop = self.start_stop[:, i]

            baseline      = baseline_buffer[start:stop, :]

            shape = baseline.shape
            baseline = baseline.reshape(shape[0], 4, 64)
            baseline = baseline.reshape(self.Nfeed, tod_lens[i], 4, 64)
            baseline = baseline.transpose(0, 2, 3, 1)
            baseline[:, 0, :, :] = baseline[:, 0, ::-1, :]
            baseline[:, 2, :, :] = baseline[:, 2, ::-1, :]

            baseline = baseline.astype(np.float32)
            
            new_name = self.names[i].split(".")
            new_name = new_name[0] + "_temp." + new_name[1]
            
            infile = h5py.File(outfile_path + new_name, "w")
            infile.create_dataset("tod_baseline", data = baseline, dtype = "float32")
            infile.close()

    def save_baseline_tod_per_freq(self, sb, freq, baseline_buffer):
        tod_lens = self.tod_lens
        tod_lens = tod_lens[::self.Nfeed]

        #outfile_path = self.infile_path + "baselines/"
        #outfile_path = self.infile_path + "all_in_one/"
        #outfile_path = self.infile_path + "batch_of_third2/"
        #outfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/sim/highpass/002Hz/default/XL_dataset/co6/baselines/"
        #outfile_path = self.infile_path + "null_test/"
        outfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/wo_sim/highpass/002Hz/default/large_dataset/masked/baselines/all_in_one/"
        
        #outfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/wo_sim/highpass/002Hz/default/large_dataset/masked/baselines/batch_of_third2/"
        
        print("Saveing baselines to:", outfile_path)
        if not os.path.exists(outfile_path):
            os.mkdir(outfile_path)
        
        for i in range(len(self.names)):
            start, stop = self.start_stop[:, i]
            baseline      = baseline_buffer[start:stop]
            #print(start, stop)
            baseline = baseline.reshape(self.Nfeed, tod_lens[i])
         
            baseline = baseline.astype(np.float32)
            
            new_name = self.names[i].split(".")
            new_name = new_name[0] + "_temp." + new_name[1]
            
            outfile = h5py.File(outfile_path + new_name, "a")
            if "tod_baseline" not in outfile.keys():
                outfile.create_dataset("tod_baseline", (18, 4, 64, baseline.shape[1]), dtype = "float32")
            
            data = outfile["tod_baseline"]
            if sb == 0 or sb == 2:
                data[:-1, sb, -(freq + 1), :] = baseline
            else:
                data[:-1, sb, freq, :] = baseline

            outfile.close()

    def save_baseline_tod_per_batch(self):
        feed, sb, freq = self.feed, self.sb, self.freq
        #currentNames, tod_lens, baseline_buffer = self.currentNames, self.tod_lens, self.baseline_tod
        currentNames, tod_lens = self.currentNames, self.tod_lens
        start_stop  = self.start_stop
        #tod_lens = self.tod_lens

        #outfile_path = self.infile_path + "splittest/"
        #outfile_path = self.infile_path + "splittest2/"
        #outfile_path = self.infile_path + "feed_separated_all_in_one/"
        
        #outfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/wo_sim/highpass/002Hz/default/large_dataset/masked/baselines/splittest2/"
        #outfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/wo_sim/highpass/002Hz/default/large_dataset/masked/baselines/feed_separated_all_in_one/"
        outfile_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/sim2/default/large_dataset/masked/co6/splittest3/"

        #print("Saveing baselines to:", outfile_path)
        if not os.path.exists(outfile_path):
            os.mkdir(outfile_path)
        
        Nperbaselines = self.Nperbaselines[1:]

        for i in range(len(currentNames)):
            alive = self.feeds_alive[currentNames[i]]
            feed_idx = np.where(alive == feed)[0]

            if feed_idx.size == 1:
                amplitude = self.a[self.scan_per_baseline == currentNames[i]]
                Nperbaseline = Nperbaselines[self.scan_per_baseline == currentNames[i]]

                new_name = currentNames[i].split(".")
                new_name = new_name[0] + "_temp." + new_name[1]
                
                outfile = h5py.File(outfile_path + new_name, "a")

                if "amplitudes" not in outfile.keys() and "Nperbaseline" not in outfile.keys():
                    outfile.create_dataset("amplitudes", (18, 4, 64, amplitude.shape[0]), dtype = "float32")
                    outfile.create_dataset("Nperbaseline", (18, 4, 64, Nperbaseline.shape[0]), dtype = "float32")
                
                data_amplitudes = outfile["amplitudes"]
                data_Nperbaseline = outfile["Nperbaseline"]

                data_amplitudes[feed_idx[0], sb, freq, :] = amplitude
                data_Nperbaseline[feed_idx[0], sb, freq, :] = Nperbaseline

                outfile.close()

                """ti = time.time()
                start, stop = start_stop[:, i]

                baseline      = baseline_buffer[start:stop]
            
                baseline = baseline.astype(np.float32)
                
                new_name = currentNames[i].split(".")
                new_name = new_name[0] + "_temp." + new_name[1]
                
                outfile = h5py.File(outfile_path + new_name, "a")
                if "tod_baseline" not in outfile.keys():
                    outfile.create_dataset("tod_baseline", (18, 4, 64, baseline.shape[0]), dtype = "float32")
                
                data = outfile["tod_baseline"]

                if data.shape[-1] != baseline.shape[-1]:
                    print("Wrong shape!", data.shape, baseline.shape, start, stop, stop - start, tod_lens[i], baseline_buffer[start:stop].shape, baseline_buffer.shape, i, len(self.currentNames), self.freq_idx)
            
                data[feed_idx[0], sb, freq, :] = baseline
                
                outfile.close()"""
                #print("Save time iterration:", time.time() - ti, "sec")
            else:
                continue

if __name__ =="__main__":
    #datapath    = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level2/Ka/sim/dynamicTsys/co6/"
    #paramfile = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_test_co6.txt"
    #paramfile = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_test_wosim_co6.txt"
    
    #paramfile = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_noise_weighted_test_co6.txt"
    #paramfile = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_noise_weighted_test_wosim_co6.txt"

    #paramfile0 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_noise_weighted_python_wohighpass_co6.txt"
    #paramfile1 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_noise_weighted_wohighpass_wosim_co6.txt"
    
    # -------------------------------------
    # Multiple different baseline lengths:
    # -------------------------------------
    
    #paramfile0 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_001s_co6.txt"
    #paramfile1 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_1s_co6.txt"
    #paramfile2 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_10s_co6.txt"
    #paramfile3 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_25s_co6.txt"
    #paramfile4 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_50s_co6.txt"
    #paramfile5 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_100s_co6.txt"
    
    
    #paramfile01 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_001s_wosim_co6.txt"
    #paramfile1 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_1s_wosim_co6.txt"
    #paramfile2 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_10s_wosim_co6.txt"
    #paramfile3 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_25s_wosim_co6.txt"
    #paramfile4 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_50s_wosim_co6.txt"
    #paramfile5 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_100s_wosim_co6.txt"
    
    # -------------------------------------
    # Coadded obsID destriper maps:
    # -------------------------------------
    

    #paramfile0 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_obsID_10s_co6.txt"
    #paramfile1 = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_obsID_10s_wosim_co6.txt"


    #paramfiles = [paramfile0, paramfile1, paramfile2, paramfile3, paramfile4]
    #paramfiles = [paramfile0, paramfile01]
    
    #paramfiles = [paramfile0, paramfile1]  
    #paramfiles  = [paramfile1]
   
    #paramfiles = ["/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_10s_wosim_large_dataset_co6.txt"]
    #paramfiles = ["/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_destriper_10s_large_dataset_co6.txt"]
    #paramfiles = ["/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_liss_CES_mix_freqmask_co6.txt"]
    paramfiles  = ["/mn/stornext/d16/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/param_save_baseline_test_co6.txt"]
    
    eps      = 0
    #freq_idx = [113]
    freq_idx = range(4 * 64)
    N_proc = 48
    
    for pfile in paramfiles:
        t = time.time()
        destr = Destriper(param_file = pfile)
        print("Loading data and initializing pointing:")
        t0 = time.time()
        destr.get_data()
        destr.initialize_P_and_F()
        print("Loading time:", time.time() - t0, "sec")
        
        t0 = time.time()
        print("Looping over frequencies:")

        def dummy(idx):
            print("\n", "Processing frequency number:", idx, "\n")
            t = time.time()
            destr.run(freq_idx = idx)

            destr.make_map()
            print("\n", "Making map: ", time.time() - t, "sec \n")

            destr.get_rms()
            print("\n", "Making rms map: ", time.time() - t, "sec \n")

            destr.get_hits()
            print("\n", "Making hit map: ", time.time() - t, "sec \n")

            maps = np.array([destr.m, destr.rms, destr.hits])
            maps = np.where(np.isnan(maps) == False, maps, 0)
            return np.array([destr.m, destr.rms, destr.hits])
    
        with multiproc.Pool(processes = N_proc) as pool:
            full_map = pool.map(dummy, freq_idx)
        pool.close()
        pool.join()
        print("Finished frequency loop:", time.time() - t0, "sec")
    
        print("Formating output:")
        
        full_map = np.array(full_map)

        full_rms = full_map[:, 1, :, :]
        full_hits = full_map[:, 2, :, :]
        full_map = full_map[:, 0, :, :]

        full_map = full_map.reshape(4, 64, 120, 120)
        full_hits = full_hits.reshape(4, 64, 120, 120)
        full_rms = full_rms.reshape(4, 64, 120, 120)
        
        
        full_map = full_map.transpose(0, 1, 3, 2)
        full_hits = full_hits.transpose(0, 1, 3, 2)
        full_rms = full_rms.transpose(0, 1, 3, 2)
        
        print("Writing to file:")
        destr.write_map(full_map, full_hits, full_rms)    
    
    """
    map         = np.ma.masked_where(full_hits < 1, full_map)
    hits        = np.ma.masked_where(full_hits < 1, full_hits)
    rms         = np.ma.masked_where(full_hits < 1, full_rms)

    map = map[0, 0, :, :]
    hits = hits[0, 0, :, :]
    rms = rms[0, 0, :, :]

    cmap_name = "CMRmap"
    cmap = copy.copy(plt.get_cmap(cmap_name))
    cmap.set_bad("0.8", 1)

    fig0, ax0 = plt.subplots(figsize = (8, 7))


    im0 = ax0.imshow(map * 1e6, cmap = cmap, vmin = -5000, vmax = 5000)
    #im0 = ax0.imshow(hits, cmap = cmap)#, vmin = -5000, vmax = 5000)
    #im0 = ax0.imshow(rms * 1e6, cmap = cmap, vmin = 0, vmax = 1e4)
    im0.set_rasterized(True)
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig0.colorbar(im0, ax=ax0, cax = cax0)
    ax0.set_title("Destriper")
    cbar0.set_label(r"$\mu K$")
    fig0.tight_layout()
    plt.savefig("Destriper.png")
    """
    """
    eps      = 0
    freq_idx = range(4 * 64)
        
    for pfile in paramfiles:
        t = time.time()
        destr = Destriper(eps, param_file = pfile, obsID_map = True)
        print("Loading data and initializing pointing:")
        
        map  = np.zeros((len(freq_idx), 120, 120, destr.nobsIDs))
        hits = np.zeros((len(freq_idx), 120, 120, destr.nobsIDs))
        rms  = np.zeros((len(freq_idx), 120, 120, destr.nobsIDs))
        
        print("Looping over obsIDs:")
        tobs = time.time()
        for i, ID in enumerate(destr.obsIDs):
            t0 = time.time()
            destr.obsID = ID
            destr.get_data()
            destr.initialize_P_and_F()
            print("Loading time:", time.time() - t0, "sec")
        
            t0 = time.time()
            print("Looping over frequencies:", destr.obsID)
            
            def dummy(idx):
                #print("Processing frequency number:", idx, "\n")
                destr.run(freq_idx = idx)
                #destr.get_destriped_map()        
                #destr.get_noise_weighted_map(True, 0.02)        
                destr.make_map()
                destr.get_rms()
                destr.get_hits()
                maps = np.array([destr.m, destr.rms, destr.hits])
                maps = np.where(np.isnan(maps) == False, maps, 0)
                return np.array([destr.m, destr.rms, destr.hits])

            with multiproc.Pool(processes = 48) as pool:

                obs_map = pool.map(dummy, freq_idx)
            
            pool.close()
            pool.join()
            
            print("Finished frequency loop:", time.time() - t0, "sec")

            print("Formating output:")
        
            obs_map = np.array(obs_map)
        
            map[..., i]  = obs_map[:, 0, :, :]
            rms[..., i]  = obs_map[:, 1, :, :]
            hits[..., i] = obs_map[:, 2, :, :]
        
        print(map.shape)
        map  = map.reshape( 4, 64, 120, 120, destr.nobsIDs)
        hits = hits.reshape(4, 64, 120, 120, destr.nobsIDs)
        rms  = rms.reshape( 4, 64, 120, 120, destr.nobsIDs)

        map         = np.ma.masked_where(hits < 1, map)
        hits        = np.ma.masked_where(hits < 1, hits)
        rms         = np.ma.masked_where(hits < 1, rms)
        
        map  = map.transpose(0, 1, 3, 2, 4)
        hits = hits.transpose(0, 1, 3, 2, 4)
        rms  = rms.transpose(0, 1, 3, 2, 4)

        
        var_inv = 1 / rms ** 2
        
        map = np.nansum(map * var_inv, axis = -1) / np.nansum(var_inv, axis = -1)     
        rms = 1 / np.sqrt(np.nansum(var_inv, axis = -1))
        hits = np.nansum(hits, axis = -1)
        print("max, min of hits:", np.nanmax(hits), np.nanmin(hits))
        
        print(map.shape)
    
        print("Finished obsID loop:", time.time() - tobs, "sec")

        map         = np.where(np.isnan(hits) == False, map, 0)
        hits        = np.where(np.isnan(hits) == False, hits, 0)
        rms         = np.where(np.isnan(hits) == False, rms, 0)
        
        print("Writing to file:")
        destr.write_map(map, hits, rms)    

    """        
    """
    map_obsID = map[0, 0, :, :]
    hits_obsID = hits[0, 0, :, :]
    rms_obsID = rms[0, 0, :, :]

    cmap_name = "CMRmap"
    cmap = copy.copy(plt.get_cmap(cmap_name))
    cmap.set_bad("0.8", 1)
    fig0, ax0 = plt.subplots(figsize = (8, 7))
    
    
    im0 = ax0.imshow(map_obsID * 1e6, cmap = cmap, vmin = -5000, vmax = 5000)
    #im0 = ax0.imshow(hits_obsID, cmap = cmap)#, vmin = -5000, vmax = 5000)
    #im0 = ax0.imshow(rms_obsID * 1e6, cmap = cmap, vmin = 0, vmax = 1e4)
    im0.set_rasterized(True)
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig0.colorbar(im0, ax=ax0, cax = cax0)
    ax0.set_title("Destriper")
    cbar0.set_label(r"$\mu K$")
    fig0.tight_layout()
    plt.savefig("DestriperObsID.png")
    """

    

    