
// Compile as
// g++ -shared -o whitenoiselib.so.1 whitenoiselib.cpp -std=c++11 -O3 -fPIC

#include <stdio.h>
#include <random>
#include <iostream>
#include <ctime>
#include <cmath>


void whitenoise(float* Tsys,    float* noise,   float dt,   float dnu,
                int n0,         int n1,         int n2,     int n3){
    
    int prod = n2 * n3;     // Precomputing index factors for flattened index
    int prod1 = prod * n1;
    unsigned long idx;                // Index of flattened arrays
    
    unsigned long i, j, k, l;
    float sigma_T;
    float sqrt_factor = std::sqrt(dt * dnu * 1e9);

    std::mt19937_64 generator;
    generator.seed(time(NULL));

    for (i = 0; i < n0; i++){
        for (j = 0; j < n1; j++){
            for (k = 0; k < n2; k++){
                for (l = 0; l < n3; l++){
                    idx = prod1 * i + prod * j + n3 * k + l;
                    
                    sigma_T = Tsys[idx] / sqrt_factor;
                    std::normal_distribution<double> distribution(0.0 , sigma_T);
                    
                    noise[idx] = distribution(generator);

                }
            }
        }
    }
}