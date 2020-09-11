import numpy as np 
import ctypes 
import sys

np.random.seed(12323)

px_indx = np.zeros(50, dtype = ctypes.c_int)
px_indx += np.random.randint(0, 12 * 12, len(px_indx))

tod     = np.zeros((5, 5, 50), dtype = ctypes.c_float)
tod     += np.random.normal(1, 2, tod.shape)

map = np.zeros((5, 5, 12 * 12), dtype = ctypes.c_float)
nhit = np.zeros((5, 5, 12 * 12), dtype = ctypes.c_int)

nsb, nfreq, ntod = tod.shape
nsb, nfreq, nbin = map.shape

tod_idx = np.zeros(len(map.flatten()), dtype = ctypes.c_int)

print("Before: ")
print(px_indx)
print(tod[4, 3, :])


maputilslib = ctypes.cdll.LoadLibrary("histutils.so.1")  # Load shared C utils library.
float32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=3, flags="contiguous")   # 4D array 32-bit float pointer object.
int32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=3, flags="contiguous")       # 4D array 32-bit integer pointer object.
int32_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, flags="contiguous")       # 4D array 32-bit integer pointer object.

maputilslib.histogram.argtypes = [int32_array1, float32_array3, float32_array3, int32_array3,        # Specifying input types for C library function.
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

maputilslib.histogram(px_indx, tod, map, nhit, 
                        nsb, nfreq, ntod, nbin)

"""
#idx0 = np.arange(len(tod.flatten()))
for j in range(nsb):
    for k in range(nfreq):
        for l in range(nbin):
            idx = j * nfreq * nbin + k * nbin + l
            print(tod_idx[idx], idx)
sys.exit()
"""

print("After: ")
map = map.reshape(5, 5, 12, 12)
nhit = nhit.reshape(5, 5, 12, 12)

print(map[4, 3, :, :])
print(nhit[4, 3, :, :])
print(np.any(nhit != 0), np.where(nhit != 0))
print("Numpy: ")

hist, edges = np.histogram(px_indx, bins = nbin, range = (0, nbin), weights = tod[4, 3, :])
hits, edges = np.histogram(px_indx, bins = nbin, range = (0, nbin))
hist = hist.reshape(12, 12)
hits = hits.reshape(12, 12)

print(hist)
print(hits)
print(np.allclose(hist, map[4, 3, :, :]))
print(np.allclose(hits, nhit[4, 3, :, :]))
