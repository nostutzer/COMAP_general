import numpy as np
import h5py
import time
import shutil


class cube2map:
    def __init__(self, cube_filename, map_in_filename, map_out_filename):
        self.cube_filename = cube_filename          # Filepath of simulated cube.
        self.map_in_filename = map_in_filename      # Filepath of already existing level1 file.
        self.map_out_filename = map_out_filename    # Filepath of simulated output level1 file.
        
    def run(self):
        print("Loading Cube"); 
        t0 = time.time()
        self.load_cube()
        print("Time: ", time.time()-t0, " sec")
        print("Copying outfile"); 
        t0 = time.time()
        self.make_outfile()
        print("Time: ", time.time()-t0, " sec"); 
        t0 = time.time()
        print("Saving cube to map file")
        self.save_outfile()
        print("Time: ", time.time()-t0, " sec"); 
        
    def load_cube(self):
        """
        Read the simulated datacube into memory.
        """
        cube = np.load(cube_filename)
        cubeshape = cube.shape
        cube = cube.reshape(cubeshape[0], cubeshape[1], 4, 1024)  # Flatten the x/y dims, and split the frequency (depth) dim in 4 sidebands.
        cube = cube.reshape(cubeshape[0], cubeshape[1], 4, 64, 16)
        cube = np.mean(cube, axis = -1)     # Averaging over 16 frequency channels
        cube = cube.transpose(2, 3, 0, 1)        # Reorder dims such that the x/y dim is last, and the frequencies first (easier to deal with later).
        cube /= np.max(cube)     # Normalizing cube so that the max value is 1 K
        cube -= np.mean(cube, axis = (0, 1))    # Normalizing cube so that the max value is 1 K
        self.cube = cube

    def make_outfile(self):
        """
        Create a copy of the input level1 file, such that we can simply replace the map with simulated data later.
        """
        shutil.copyfile(self.map_in_filename, self.map_out_filename)
    
    def save_outfile(self):
        with h5py.File(self.map_out_filename, "r+") as outfile:
            map     = outfile["map"]
            rms     = outfile["rms"]
            nhit    = outfile["nhit"]
            map[...]    = np.zeros_like(map)
            rms[...]    = np.zeros_like(rms)
            nhit[...]   = np.zeros_like(nhit)
            
            try:
                map_coadd   = outfile["map_coadd"]
                rms_coadd   = outfile["rms_coadd"]
                nhit_coadd  = outfile["nhit_coadd"]
                map_coadd[...]  = self.cube
                rms_coadd[...]  = np.ones(rms_coadd.shape)
                nhit_coadd[...] = np.ones(nhit_coadd.shape)

            except KeyError:
                map_beam   = outfile["map_beam"]
                rms_beam   = outfile["rms_beam"]
                nhit_beam  = outfile["nhit_beam"]
                map_beam[...]  = self.cube
                rms_beam[...]  = np.ones(rms_beam.shape)
                nhit_beam[...] = np.ones(nhit_beam.shape)

if __name__ == "__main__":
    cube_path = "/mn/stornext/d16/cmbco/comap/protodir/"
    cube_filename = cube_path + "cube_real.npy"
    
    map_in_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/sim/"
    map_in_filename = map_in_path + "co6_map.h5"

    map_out_path = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/sim/"
    map_out_filename =  map_out_path + "co6_cube_map.h5"

    cube2map = cube2map(cube_filename, map_in_filename, map_out_filename)
    cube2map.run()