// Compile as
// gcc -shared -o histutils.so.1 histutils.c -std=c11 -O3 -fPIC

#include <stdio.h>
#include <omp.h>
#include <math.h>

/*
void histogram1D(int* px_idx, float* tod, float* map, int* nhit, int nbin, int ntod){
    int i, j;
    for (i = 0; i < nbin; i++){
        for (j = 0; j < ntod; j++){
            if (px_idx[j] == i){
                map[i] += tod[j];
                nhit[i] ++;

            }
        }
    }
}
*/

void histogram(int* px_idx, float* tod, float* map, int* nhit, 
               int nsb, int nfreq, int ntod, int nbin){
        
    int prod = nfreq * ntod;     // Precomputing index factors for flattened index
    unsigned long idx;                    // Index of flattened arrays
    
    int prod_bin = nfreq * nbin;     // Precomputing index factors for flattened index
    unsigned long idx_bin;                    // Index of flattened arrays
    
    int i, j, k;

    for (i = 0; i < nsb; i++){
        for (j = 0; j < nfreq; j++){
            for (k = 0; k < ntod; k ++){
                idx             = prod * i + ntod * j + k;
                idx_bin         = prod_bin * i + nbin * j + px_idx[k];
                if (isnan(tod[idx]) == 0){
                    map[idx_bin]   += tod[idx];
                }
                nhit[idx_bin]  ++;
            }
        }
    }    
}

void nhits(int* px_idx, int* nhit, int nsb, 
            int nfreq, int ntod, int nbin){
        
    int prod = nfreq * nbin;     // Precomputing index factors for flattened index
    unsigned long idx;                    // Index of flattened arrays
    
    int i, j, k;
    
    for (i = 0; i < nsb; i++){
        for (j = 0; j < nfreq; j++){
            for (k = 0; k < ntod; k ++){
                idx         = prod * i + nbin * j + px_idx[k];
                nhit[idx]  ++;
            }
        }
    }    
}
