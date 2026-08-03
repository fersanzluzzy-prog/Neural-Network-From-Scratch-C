#include "network.h"
#include <stdio.h>
#include <stdlib.h>

int crearRed(struct network **resultado){
    struct network *res = malloc(sizeof(struct network));
    if(res == NULL){return -1;}

    res->capas = NULL;
    res->numeroCapas = 0;
    *resultado = res;
    return 1;
}

int añadirCapa(struct layer *capa, struct network *red){
    if(red->capas == NULL){
        red->capas = malloc(sizeof(struct layer *));
        red->capas[0] = capa;
        red->numeroCapas++;
        return 1;
    }
    else{
        int n = red->numeroCapas;
        int nEntradasCapaNueva = capa->nEntradas;
        int nSalidasUltimaCapa = red->capas[n-1]->nSalidas;
        if(nEntradasCapaNueva == nSalidasUltimaCapa){
            struct layer **temporal = realloc(red->capas, n * sizeof(struct layer *));
            if(temporal != NULL){
                red->capas = temporal;
                red->capas[n] = capa;
                red->numeroCapas++;
                return 1;
            }
            return -1;
        }
        return -1;
    }
}

void eliminarRed(struct network **red){
    int n = (*red)->numeroCapas;
    for(int i=0; i<n; i++){
        eliminarLayer(&(*red)->capas[i]);
    }
    free((*red)->capas);
    *red = NULL;
}


