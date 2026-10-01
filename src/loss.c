#include <math.h>
#include <stdlib.h>

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
    return 1; 
}

//resultados reales - resultados esperados
int mseDerivada(struct matrix reales, struct matrix esperadas, struct matrix *resultado){
    if(reales.col == esperadas.col && reales.fil == esperadas.fil){
        int nDatos = reales.fil * reales.col;
        for (int i=0; i<nDatos; i++){
            resultado->datos[i] = (-esperadas.datos[i] + reales.datos[i])/nDatos;
        }
        return 0;
    }
    return 1;  
}






