#include "layer.h"
#include <stdio.h>
#include <stdlib.h>

//si devuelven int, 1 si va todo bien, -1 si ocurre algun error

int crearLayer(struct matrix *pesos, struct matrix *bias, struct layer **resultado){
    *resultado = malloc(sizeof(struct layer));

    if(*resultado==NULL){
        printf("Error al reservar memoria creando capa.");
        return -1;
    }

    if(pesos->fil == bias->fil || bias->col == 1){
        (*resultado)->nSalidas = bias->fil;
        (*resultado)->nEntradas = pesos->col;
        (*resultado)->pesos = pesos;
        (*resultado)->bias = bias;
        return 1;
    }
    return -1;
}

void eliminarLayer(struct layer **layer){
    free((*layer)->pesos);
    free((*layer)->bias);
    free(*layer);
    *layer = NULL;
}

int forward(struct matrix *entradas, struct layer *layer, struct matrix *salidas){
    if (entradas->fil == layer->nEntradas && salidas->fil == layer->nSalidas){
        multiplicacionMatricial(layer->pesos, entradas, salidas);
        struct matrix copia = *salidas; 
        suma(copia, *layer->bias, salidas);
        
        return 1;
    }
    else{return -1;}
}









