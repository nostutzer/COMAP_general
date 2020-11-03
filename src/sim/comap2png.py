import time
t0 = time.time()
import numpy as np
import h5py
from tqdm import trange
import argparse
import ctypes
t1 = time.time()
print("Imports: %.4f" % (t1-t0))

USE_CTYPES = False
USE_GNUPLOT = False

class COMAP2PNG:
    def __init__(self, from_commandline=True, filename="", feeds=range(1,20), sidebands=range(1,4), frequencies=range(1,65), maptype="map", outname="outfile", outpath="", plottype="png"):

        self.avail_maps = ["map", "rms", "map_rms", "sim", "rms_sim", "hit", "feed", "var"]
        self.avail_plottypes = ["png", "gif", "mp4"]
        self.sideband_names = ["A:LSB", "B:LSB", "A:USB", "B:USB"]

        if from_commandline:
            parser = argparse.ArgumentParser()
            parser.add_argument("filename", type=str)
            parser.add_argument("-d", "--detectors", type=str, default="range(1,20)", help="List of detectors(feeds), on format which evals to Python list or iterable, e.g. [1,4,9] or range(2,6).")
            parser.add_argument("-s", "--sidebands", type=str, default="range(1,5)", help="List of sidebands, on format which evals to Python list or iterable, e.g. [1,2], [3], or range(1,3).")
            parser.add_argument("-f", "--frequencies", type=str, default="range(1,65)", help="List of frequencies, on format which evals to Python list or iterable, e.g. [34,36,41], [43], or range(12,44). Note that if you specify a frequency, a single sideband must be selected.")
            parser.add_argument("-m", "--maptype", type=str, default="map")
            parser.add_argument("-o", "--outname", type=str, default="outfile")
            parser.add_argument("-p", "--outpath", type=str, default="")
            parser.add_argument("-t", "--plottype", type=str, default="png", help="Choose from png, gif, mp4.")
            args = parser.parse_args()
            try:
                self.feeds       = np.array(eval(args.detectors))
                self.sidebands   = np.array(eval(args.sidebands))
                self.frequencies = np.array(eval(args.frequencies))
            except:
                raise ValueError("Could not resolve detectors, sidebands, or frequencies as a Python iterable.")
            self.filename   = args.filename
            self.maptype    = args.maptype
            self.outpath    = args.outpath
            self.outname    = args.outname
            self.plottype   = args.plottype

        else:
            self.feeds       = np.array(feeds)
            self.sidebands   = np.array(sidebands)
            self.frequencies = np.array(frequencies)
            self.maptype     = maptype
            self.outpath     = outpath
            self.outname     = outname
            self.filename    = filename
            self.plottype    = plottype
            if len(filename) < 0:
                raise ValueError("You must provide an input filename.")
            
        self.parse_arguments()


    def parse_arguments(self):
        
        if not self.maptype in self.avail_maps:
            raise ValueError("Don't recognize map type %s. Available types are %s" % (self.maptype, str(self.avail_maps)))

        if (self.feeds < 1).any() or (self.feeds > 19).any():
            raise ValueError("Feeds must be in range 1-19.")
        if (self.sidebands < 1).any() or (self.sidebands > 4).any():
            raise ValueError("Sidebands must be in range 1-4.")
        if (self.frequencies < 1).any() or (self.frequencies > 64).any():
            raise ValueError("Frequencies must be in range 1-64.")
    
        if len(self.frequencies) != 64:
            if len(self.sidebands) != 1:
                raise ValueError("If you specify frequencies, you must specify a single sideband.")
            
        self.indexing = []
        non_continuous = 0
        for item in [self.feeds, self.sidebands, self.frequencies]:
            if (len(item) > 1) and ((item[1:] - item[:-1]) != 1).any():  # If there are gaps in the list.
                non_continuous += 1
                self.indexing.append(item-1)  # Data is 0-indexed, input is 1 indexed. Subtract 1.
            else:
                self.indexing.append(slice(item[0]-1, item[-1]))  # Same as above.
        self.indexing = tuple(self.indexing)  # Tuples are nice for slicing.

        if len(self.feeds) == 19 and not self.maptype == "feed":
            # If we want to average over all feeds, and also don't want a "seenbyfeed" map, we can use the _beam datasets instead of the full ones.
            self.all_feeds = True
        else:
            self.all_feeds = False

                
        if non_continuous > 1:
            raise ValueError("At most one of detectors, sidebands and frequencies may have gaps in their values (gap meaning ex: [1,2,3,6].)")
        

    def run(self):
        t0 = time.time()
        self.read_h5()
        t1 = time.time()
        self.make_maps()
        t2 = time.time()
        self.plot_maps()
        t3 = time.time()
        print("Reading h5 files: %.4f" % (t1-t0))
        print("Calculating maps: %.4f" % (t2-t1))
        print("Writing map to png: %.4f" % (t3-t2))


    def read_h5(self):
        h5file    = h5py.File(self.filename, "r")
        self.nx   = h5file["n_x"][()]
        self.ny   = h5file["n_y"][()]
        self.x    = h5file["x"][()]
        self.y    = h5file["y"][()]
        self.freq = h5file["freq"][()]

        if self.all_feeds:  # If we're reading all feeds, we can use the "beam" dataset.
            try:
                self.map_full = h5file["map_beam"][self.indexing[1:]][None,:,:,:,:]  # Beam doesn't contain first index, so skip it,
                self.rms_full = h5file["rms_beam"][self.indexing[1:]][None,:,:,:,:]  # Then add empty feed dim,
                self.hit_full = h5file["nhit_beam"][self.indexing[1:]][None,:,:,:,:] # easier compatability with later code.
            except KeyError:
                self.map_full = h5file["map_coadd"][self.indexing[1:]][None,:,:,:,:]  # Beam doesn't contain first index, so skip it,
                self.rms_full = h5file["rms_coadd"][self.indexing[1:]][None,:,:,:,:]  # Then add empty feed dim,
                self.hit_full = h5file["nhit_coadd"][self.indexing[1:]][None,:,:,:,:] # easier compatability with later code.
        else:
            self.map_full = h5file["map"][self.indexing]
            self.rms_full = h5file["rms"][self.indexing]
            self.hit_full = h5file["nhit"][self.indexing]

        self.num_feeds, self.num_bands, self.num_freqs, self.nx, self.ny = self.map_full.shape

    
    def make_maps(self):
        
        map_full, rms_full, hit_full = self.map_full, self.rms_full, self.hit_full
        nfeed, nband, nfreq, nx, ny = self.map_full.shape


        if USE_CTYPES:  # Ctypes implementation (much faster).
            map_out = np.zeros((ny,nx), dtype=np.float32)
            rms_out = np.zeros((ny,nx), dtype=np.float32)
            hit_out = np.zeros((ny,nx), dtype=np.int32)
            
            maplib = ctypes.cdll.LoadLibrary("maplib.so.1")  # Load shared library
            float32_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=2, flags="contiguous")
            float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")
            int32_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=2, flags="contiguous")
            int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")
            maplib.makemaps.argtypes = [float32_array5, float32_array5, int32_array5,
                                        float32_array2, float32_array2, int32_array2,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            maplib.makemaps(map_full, rms_full, hit_full, map_out, rms_out, hit_out, nfeed, nband, nfreq, nx, ny)


        else:  # Pure Numpy version.
            nx, ny = self.nx, self.ny
            if self.plottype == "png":  # If we're plotting a png, we sum over sidebands/freqs.
                sum_axis = (0,1,2)      # If we're plotting a movie or gif, we don't.
            else:
                sum_axis = (0)

            if self.maptype == "var":
                # self.var_out = np.zeros((nx,ny))
                # for x in range(nx):
                #     for y in range(ny):
                #         tempdata = []
                #         for i in range(nfeed):
                #             for j in range(nband):
                #                 for k in range(nfreq):
                #                     if rms_full[i,j,k,x,y] != 0:
                #                         tempdata.append(map_full[i,j,k,x,y]/rms_full[i,j,k,x,y])
                #         self.var_out[x,y] = np.var(tempdata)
                self.var_out = np.var(map_full/np.where(rms_full != 0, rms_full, 1.0), axis=sum_axis)                
            else:
                inv2_hit_rms_full = np.where(hit_full != 0, np.divide(1.0, rms_full**2, out=np.zeros_like(rms_full), where=rms_full!=0), 0.0)
                self.map_out = np.nansum(map_full*inv2_hit_rms_full, axis=sum_axis)
                self.rms_out = np.nansum(inv2_hit_rms_full, axis=sum_axis)
                self.hit_out = np.nansum(hit_full, axis=sum_axis)

                self.map_out = self.map_out/np.where(self.rms_out==0, np.inf, self.rms_out)
                self.rms_out = np.sqrt(1.0/np.where(self.rms_out==0, np.inf, self.rms_out))

        if self.maptype == "feed":
            if self.plottype == "png":
                hitbyfeed = np.sum(hit_full, axis=(1,2))
            elif self.plottype in ["gif", "mp4"]:
                hitbyfeed = hit_full
            self.feed_out = np.sum(hitbyfeed > 0.01*self.hit_out, axis=0)

    
    def plot_maps(self):
        x_lim, y_lim, color_lim = [None,None], [None,None], [None,None]

        if self.maptype == "map":
            plotdata = self.map_out*1e6
            color_lim[1] = 1*np.std(plotdata)
            color_lim[0] = -color_lim[1]
            print("Portion of map inside crange: %.3f" % (np.sum(np.abs(plotdata) < color_lim[1])/np.size(plotdata)))
        elif self.maptype == "rms":
            plotdata = self.rms_out*1e6
            color_lim = 0, 1*np.std(plotdata)
            print("Portion of map inside crange: %.3f" % (np.sum(np.abs(plotdata) < color_lim[1])/np.size(plotdata)))
        elif self.maptype == "hit":
            plotdata = self.hit_out
            color_lim = np.min(plotdata), np.max(plotdata)
        elif self.maptype == "map_rms":
            plotdata = self.map_out/self.rms_out
        elif self.maptype == "var":
            plotdata = self.var_out
        elif self.maptype == "feed":
            plotdata = self.feed_out
        # elif self.maptype == "sim":
        #     plotdata = self.sim_out
        # elif self.maptype == "sim_rms":
        #     plotdata = self.simrms_out
        plotdata = np.ma.masked_where(self.hit_out < 1, plotdata)
        # plotdata[self.hit_out < 1] = np.nan

        x, y = self.x, self.y
        dx = x[1] - x[0]
        x_lim[0] = x[0] - 0.5*dx; x_lim[1] = x[-1] + 0.5*dx
        dy = y[1] - y[0]
        y_lim[0] = y[1] - 0.5*dy; y_lim[1] = y[-1] + 0.5*dy

        if USE_GNUPLOT:
            import PyGnuplot as gp
            
            gp.s(plotdata)
            gp.c('set term png')
            gp.c('set output "%s"' % (self.outpath + self.outname + ".png"))
            if color_lim[0] is not None and color_lim[1] is not None:
                gp.c('set cbrange [%d:%d]' % (color_lim[0], color_lim[1]))
            gp.c('set size ratio %f' % ((y_lim[1]-y_lim[0])/(x_lim[1]-x_lim[0])))
            gp.c("set xrange[%f:%f]" % (x_lim[0], x_lim[1]))
            gp.c("set yrange[%f:%f]" % (y_lim[0], y_lim[1]))
            gp.c('set datafile missing "--"')
            gp.c('plot "tmp.dat" matrix using (%f+$1*%f):(%f+$2*%f):3 with image title ""' % (x_lim[0], dx, y_lim[0], dy))

        else:
            import matplotlib.pyplot as plt
            import matplotlib
            import copy
            matplotlib.use("Agg")  # No idea what this is. It resolves an error when writing gif/mp4.
            cmap_name = "CMRmap"
            cmap = copy.copy(plt.get_cmap(cmap_name))
            cmap.set_bad("0.8", 1) # Set color of masked elements to gray.
            fig, ax = plt.subplots()
            fig.set_figheight(5)
            fig.set_figwidth(9)
            ax.set_ylabel('Declination [deg]')
            ax.set_xlabel('Right Ascension [deg]')

            if self.plottype == "png":
                img = ax.imshow(plotdata, extent=(x_lim[0],x_lim[1],y_lim[0],y_lim[1]), interpolation='nearest',
                                    aspect='equal', cmap=cmap, origin='lower',
                                    vmin=color_lim[0], vmax=color_lim[1])

                title = self.make_title()
                ax.set_title(title)
                # ax.set_title("Sideband: %s | Channel: %d | Freq: %.3f GHz" % (self.sideband_names[s], i%64, self.freq[s,f]))
                cbar = fig.colorbar(img)
                cbar.set_label("$\mu K$")
                fig.savefig(self.outpath + self.outname + ".png")

            elif self.plottype in ["mp4", "gif"]:
                import matplotlib.animation as animation

                img = ax.imshow(plotdata[0,0], extent=(x_lim[0],x_lim[1],y_lim[0],y_lim[1]), interpolation='nearest',
                                    aspect='equal', cmap=cmap, origin='lower',
                                    vmin=color_lim[0], vmax=color_lim[1])
                fig.colorbar(img)

                if self.plottype == "mp4": # The first handful of frames stutter a lot (no idea why),
                    holdframes = 4         # so we render some static frames first (only an issue with mp4).
                else:
                    holdframes = 0

                def update(i):
                    if i < holdframes:
                        i = 0
                    else:
                        i -= holdframes
                    f = i%len(self.frequencies)
                    s = i//len(self.frequencies)
                    img.set_data(plotdata[s,f])
                    title = "Maptype: " + self.maptype + " | " + self.filename + "\n"
                    title += "Sideband %s | Channel %d | Freq %.2f GHz" % (self.sideband_names[s], i%64, self.freq[s,f])
                    ax.set_title(title)
                    return [img]
                ani = animation.FuncAnimation(fig, update, frames=len(self.frequencies)*len(self.sidebands)+holdframes, interval=200, blit=False, repeat_delay=1000)

                if self.plottype == "gif":
                    ani.save(self.outpath + self.outname + ".gif", writer="imagemagick")
                elif self.plottype == "mp4":
                    ani.save(self.outpath + self.outname + ".mp4", writer="ffmpeg")



    def make_title(self):
        title = ""
        title += "Maptype: " + self.maptype + " | "
        title += str(self.filename) + "\n"
        if len(self.feeds) == 19:
            title += "Feeds: all"
        elif len(self.feeds) == 1:
            title += "Feed: %d" % self.feeds[0]
        elif ((self.feeds[1:] - self.feeds[:-1]) == 1).all():
            title += "Feeds: %d-%d" % (self.feeds[0], self.feeds[-1])
        else:
            title += "Feeds: " + ", ".join([str(feed) for feed in self.feeds])
        title += " | "
        if len(self.sidebands) == 4:
            title += "All SB"
        elif len(self.sidebands) == 1:
            title += self.sideband_names[self.sidebands[0]-1]
        else:
            title += " + ".join([self.sideband_names[s-1] for s in self.sidebands])
        title += " | "
        if len(self.frequencies) == 64:
            title += "All ch."
        elif len(self.frequencies) == 1:
            title += "Ch.: %d | Freq: %.3f GHz" % (self.frequencies[0], self.freq[self.sidebands[0]][self.frequencies[0]])
        elif ((self.frequencies[1:] - self.frequencies[:-1]) == 1).all():
            title += "Ch.: %d-%d | Freqs: %.3f - %.3f GHz" % ((self.frequencies[0], self.frequencies[-1], self.freq[self.sidebands[0]][self.frequencies[0]], self.freq[self.sidebands[0]][self.frequencies[-1]]))
        else:
            title += "Ch.: " + ", ".join([str(freq) for freq in self.frequencies])
        return title

if __name__ == "__main__":
    map2png = COMAP2PNG()
    map2png.run()
