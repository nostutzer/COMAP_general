// Compile as
// gcc -shared -o tsyslib.so.1 tsyslib.c -std=c11 -O3 -fPIC -fopenmp

#include <stdio.h>
#include <omp.h>

void tsys_calc(float* Tsys, float* Pcold, double* Thot, double* t, double* Phot, double* Phot_t, double Tcmb,
               int _nfeed, int _nband, int _nfreq, int _ntod){
    unsigned long nfeed, nband, nfreq, ntod, feed, band, freq, tod, Ph_idx, tsys_idx;
    double t1, t2, Th1, Th2, Ph1, Ph2, t_now, Th_now, Ph_now, Pc_now;
    nfeed = (unsigned long) _nfeed;
    nband = (unsigned long) _nband;
    nfreq = (unsigned long) _nfreq;
    ntod  = (unsigned long) _ntod;
    
    #pragma omp parallel for private(feed, band, freq, tod, Ph_idx, tsys_idx, t1, t2, Th1, Th2, Ph1, Ph2, t_now, Th_now, Ph_now, Pc_now)
    
    for(feed=0; feed<nfeed; feed++){
        t1 = Phot_t[feed*2];  // Times of first and second calibration.
        t2 = Phot_t[feed*2+1];
        Th1 = Thot[feed*2];
        Th2 = Thot[feed*2+1];
        for(band=0; band<nband; band++){
            for(freq=0; freq<nfreq; freq++){
                Ph_idx = (feed*nband*nfreq + band*nfreq + freq)*2;  //*2 because Ph have two measurements per obsid.
                Ph1 = Phot[Ph_idx];
                Ph2 = Phot[Ph_idx+1];

                for(tod=0; tod<ntod; tod++){
                    tsys_idx = feed*nband*nfreq*ntod + band*nfreq*ntod + freq*ntod + tod;
                    t_now = t[tod];
                    // double Th_now = Thot[tod];
                    Th_now = (Th1*(t2 - t_now) + Th2*(t_now - t1))/(t2 - t1);
                    Ph_now = (Ph1*(t2 - t_now) + Ph2*(t_now - t1))/(t2 - t1);
                    Pc_now = Pcold[tsys_idx];
                    Tsys[tsys_idx] = (Th_now - Tcmb)/(Ph_now/Pc_now - 1.0);
                }
            }
        }
    }
}