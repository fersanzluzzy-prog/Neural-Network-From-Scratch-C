#include "training.h"


int training(struct network *red, struct matrix **entradas, struct matrix **resultadosEsperados, float factorAprendizaje, int nEntradas, int epocas){
    
    struct matrix **resultadosReales = malloc(sizeof(struct matrix *)*nEntradas);

    for(int j=0; j<epocas; j++){
        for(int i=0; i<nEntradas; i++){
            resultadosReales[i] = forwardRed(red, entradas[i]);
            backpropRed(red, *resultadosReales[i], *resultadosEsperados[i]);
            descensoDeGradiente(red, factorAprendizaje);
        }
        if(j%10 == 0){printf("Loss en época %i: %.9f\n", j, mse(*resultadosReales[nEntradas-1], *resultadosEsperados[nEntradas-1]));}
    }
    free(resultadosReales);

    return 0;
}