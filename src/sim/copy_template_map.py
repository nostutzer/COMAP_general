import numpy as np
import h5py

inname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/templates/co7_template_map_old.h5"
outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/templates/co7_template_map.h5"

infile = h5py.File(inname, "r")
outfile = h5py.File(outname, "a")

for name in infile.keys():
    condition1 = "map" != name
    condition2 = ("rms" != name) and condition1
    condition3 = ("nhit" != name) and condition2
    if "map_beam" in name:
        data = np.array(infile[name])[()]
        outfile.create_dataset("map_coadd", data = data)

    elif "rms_beam" in name:
        data = np.array(infile[name])[()]
        outfile.create_dataset("rms_coadd", data = data)
    
    elif "nhit_beam" in name:
        data = np.array(infile[name])[()]
        outfile.create_dataset("nhit_coadd", data = data)
    elif condition3:
        data = np.array(infile[name])[()]
        outfile.create_dataset(name, data = data)
infile.close()
outfile.close()
