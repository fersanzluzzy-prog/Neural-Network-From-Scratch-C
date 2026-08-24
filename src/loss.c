#include <math.h>

#include "matrix.h"

float mse(struct matrix reales, struct matrix esperadas){
    if(reales.col == esperadas.col && reales.fil == esperadas.fil){
        float resultado = 0.0;
        int nDatos = reales.fil * reales.col;
        for (int i=0; i<nDatos; i++){
            resultado += pow((esperadas.datos[i] - reales.datos[i]), 2)/2;
        }
        return resultado/nDatos;
    }
    return -1; 
}






