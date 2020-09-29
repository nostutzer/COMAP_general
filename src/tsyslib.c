// Compile as
// gcc -shared -o tsyslib.so.1 tsyslib.c -std=c11 -O3 -fPIC

#include <stdio.h>

void tsys_calc(float* Tsys, float* Pcold, double* Thot, double* t, double* Phot, double* Phot_t, double Tcmb,
               int _nfeed, int _nband, int _nfreq, int _ntod){
    unsigned long nfeed = (unsigned long) _nfeed;
    unsigned long nband = (unsigned long) _nband;
    unsigned long nfreq = (unsigned long) _nfreq;
    unsigned long ntod  = (unsigned long) _ntod;
    for(unsigned long feed=0; feed<nfeed; feed++){
        double t1 = Phot_t[feed*2];  // Times of first and second calibration.
        double t2 = Phot_t[feed*2+1];
        double Th1 = Thot[feed*2];
        double Th2 = Thot[feed*2+1];
        for(unsigned long band=0; band<nband; band++){
            for(unsigned long freq=0; freq<nfreq; freq++){
                unsigned long Ph_idx = (feed*nband*nfreq + band*nfreq + freq)*2;  //*2 because Ph have two measurements per obsid.
                double Ph1 = Phot[Ph_idx];
                double Ph2 = Phot[Ph_idx+1];

                for(unsigned long tod=0; tod<ntod; tod++){
                    unsigned long tsys_idx = feed*nband*nfreq*ntod + band*nfreq*ntod + freq*ntod + tod;
                    double t_now = t[tod];
                    // double Th_now = Thot[tod];
                    double Th_now = (Th1*(t2 - t_now) + Th2*(t_now - t1))/(t2 - t1);
                    double Ph_now = (Ph1*(t2 - t_now) + Ph2*(t_now - t1))/(t2 - t1);
                    double Pc_now = Pcold[tsys_idx];
                    Tsys[tsys_idx] = (Th_now - Tcmb)/(Ph_now/Pc_now - 1.0);
                }
            }
        }
    }
}