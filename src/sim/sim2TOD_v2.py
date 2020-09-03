import numpy as np
import h5py
import matplotlib.pyplot as plt
import WCS
import time
import shutil
from tqdm import trange
import sys 
import argparse
import re

class Sim2TOD:
    def __init__(self):#, cube_filename, tod_in_filename, tod_out_filename):
        #self.cube_filename = cube_filename  # Filepath of simulated cube.
        #self.tod_in_filename = tod_in_filename  # Filepath of already existing level1 file.
        #self.tod_out_filename = tod_out_filename  # Filepath of simulated output level1 file.
        self.nside = 120   # Number of pixels in each direction.
        self.dpix = 2.0/60.0  # Pixel resolution in degrees (here, 2 arcminutes)
        #self.fieldcent = [226, 55]  # Center position of pixel-image, in degrees of ra/dec. CO6
        #self.fieldcent = [170, 52.5]  # Center position of pixel-image, in degrees of ra/dec. CO7
        self.input()

    def run(self):
        print("Processing Parameterfile: "); t0 = time.time()
        self.read_paramfile()
        
        print("Time: ", time.time()-t0, " sec")
        print("Loading Cube: "); t0 = time.time()
        self.load_cube()
        
        print("Time: ", time.time()-t0, " sec")
        print("Copying Outfile: "); t0 = time.time()
        
        for i in range(len(self.tod_in_list)):
            self.tod_in_filename    = self.tod_in_path + self.tod_in_list[i]
            self.tod_out_filename   = self.tod_out_path + self.tod_in_list[i]
            
            self.make_outfile()
        
            print("Time: ", time.time()-t0, " sec")
            print("Loading TOD: "); t0 = time.time()        
            self.load_tod()
        
            print("Time: ", time.time()-t0, " sec")
            print("Calculating Tsys: "); t0 = time.time()        
            self.calc_tsys()
        
            print("Time: ", time.time()-t0, " sec")
            print("Writing sim-data to TOD: "); t0 = time.time()
            self.write_sim()
        
            print("Time: ", time.time()-t0, " sec")

    def input(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--parameters", type = str, 
                            help = """Full path and name to parameter file 
                                    containing all needed info to add simulations to
                                    l1 files.""")
        args = parser.parse_args()
        self.param_file = args.parameters
    
    def read_paramfile(self):
        param_file = open(self.param_file, "r")
        params = param_file.read()

        runlist_path = re.search(r"\nRUNLIST\s*=\s*'(\/.*?)'", params)
        self.runlist_path = str(runlist_path.group(1))
        
        tod_in_path = re.search(r"\nLEVEL1_DIR\s*=\s*'(\/.*?)'", params)
        self.tod_in_path = str(tod_in_path.group(1))
        
        tod_out_path = re.search(r"\nSIM_LEVEL1_DIR\s*=\s*'(\/.*?)'", params)
        self.tod_out_path = str(tod_out_path.group(1))
        
        cube_path = re.search(r"\nDATACUBE\s*=\s*'(\/.*?\.\w+)'", params)
        self.cube_filename = str(cube_path.group(1))

        runlist_file = open(self.runlist_path, "r")
        runlist = runlist_file.read()
        tod_in_list = re.findall(r"\/.*?\.\w+", runlist)
        self.tod_in_list = tod_in_list

        patch_name = re.search(r"\s([a-zA-Z0-9]+)\s", runlist)
        self.patch_name = str(patch_name.group(1))

        patch_def_path = re.search(r"\nPATCH_DEFINITION_FILE\s*=\s*'(\/.*?)'", params)
        self.patch_def_path = str(patch_def_path.group(1))

        patch_def_file = open(self.patch_def_path, "r")
        patch_def = patch_def_file.read()
        fieldcent   = re.search(rf"{self.patch_name}\s*([0-9.]+)\s*([0-9.]+)", patch_def) 
        self.fieldcent = [eval(fieldcent.group(1)), eval(fieldcent.group(2))]

        print("Patch def:", self.patch_def_path)
        print("Patch", self.patch_name)
        print("Field center", self.fieldcent)
        print("Runlist:", self.runlist_path)
        print("TOD in:", self.tod_in_path)
        print("TOD out:", self.tod_out_path)
        print("Cube:", self.cube_filename)
        print("# obsID", len(self.tod_in_list))
        print("obsID #1: ", self.tod_in_list[0])
        #sys.exit()
    
    def load_cube(self):
        """
        Read the simulated datacube into memory.
        """
        cube = np.load(cube_filename)
        cubeshape = cube.shape
        cube /= np.max(cube)
        cube = cube.reshape(cubeshape[0]*cubeshape[1], 4, 1024)  # Flatten the x/y dims, and split the frequency (depth) dim in 4 sidebands.
        cube = cube.transpose(1, 2, 0)  # Reorder dims such that the x/y dim is last, and the frequencies first (easier to deal with later).
        cube[0, :, :] = cube[0, ::-1, :]
        cube[2, :, :] = cube[2, ::-1, :]
        self.cube = cube 
    
    def make_outfile(self):
        """
        Create a copy of the input level1 file, such that we can simply replace the TOD with simulated data later.
        """
        shutil.copyfile(self.tod_in_filename, self.tod_out_filename)

    def load_tod(self):
        """
        Load the TOD and other relevant data from the level1 file into memory.
        """
        infile        = h5py.File(self.tod_in_filename, "r")

        self.tod      = np.array(infile["/spectrometer/tod"])
        #self.freqs    = np.array(infile["/spectrometer/frequency"])
        #self.tod_time = np.array(infile["/spectrometer/MJD"])
        self.feeds    = np.array(infile["/spectrometer/feeds"])
        self.nfeeds   = len(self.feeds)
        self.ra       = np.array(infile["/spectrometer/pixel_pointing/pixel_ra"])
        self.dec      = np.array(infile["/spectrometer/pixel_pointing/pixel_dec"])
        self.tod_sim  = self.tod.copy()  # The simulated data is, initially, simply a copy of the original TOD.
        #self.vane_angles    = np.array(infile["/hk/antenna0/vane/angle"])/100.0  # Degrees
        #self.vane_time      = np.array(infile["/hk/antenna0/vane/utc"])
        #self.array_features = np.array(infile["/hk/array/frame/features"])
        self.infile = infile
        infile.close()

    def calc_tsys(self):
        self.tsys = 55.0  # Hard-coded Tsys value. Needs to be calculated.

    def write_sim(self):
        nside, dpix, fieldcent, ra, dec, tod, cube, tsys, nfeeds = self.nside, self.dpix, self.fieldcent, self.ra, self.dec, self.tod, self.cube, self.tsys, self.nfeeds
        pixvec = np.zeros_like(dec, dtype = int)
        for i in trange(nfeeds):  # Don't totally understand what's going on here, it's from Håvards script.
            # Create a vector of the pixel values which responds to the degrees we send in.
            pixvec[i, :] = WCS.ang2pix([nside, nside], [-dpix, dpix], fieldcent, dec[i, :], ra[i, :])     
            # Update tod_sim values.
            #self.tod_sim[i, :, :, :] += np.nanmean(np.array(tod[i, :, :, :]), axis=2)[ :, :, None] * cube[ :, :, pixvec[i, :]] / tsys
            t = time.time()
            self.tod_sim[i, :, :, :] *= 1 + cube[ :, :, pixvec[i, :]] / tsys
            #self.tod_sim[i, :, :, :] *= 1 + cube[ :, :, pixvec[i, :]] * np.nanmax(tod)
            print("Adding to tod: ", time.time() - t, " sec")
        with h5py.File(self.tod_out_filename, "r+") as outfile:  # Write new sim-data to file.
            data = outfile["/spectrometer/tod"] 
            data[...] = self.tod_sim
        outfile.close()
if __name__ == "__main__":
    cube_path = "/mn/stornext/d16/cmbco/comap/protodir/"
    cube_filename = cube_path + "cube_real.npy"
    
    #tod_in_path = "/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2020-08/"
    #tod_in_filename = tod_in_path + "comap-0015354-2020-08-01-001323.hd5"
    tod_in_path = "/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2020-07/"
    tod_in_filename = tod_in_path + "comap-0015330-2020-07-31-040632.hd5"
    
    tod_out_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/level1/2020-07/"
    tod_out_filename = tod_out_path + "comap-0015330-2020-07-31-040632_sim_cross.hd5"

    sim2tod = Sim2TOD()#cube_filename, tod_in_filename, tod_out_filename)
    #sim2tod.input()
    #sim2tod.read_paramfile()
    sim2tod.run()
    sys.exit()