#include "network.h"
#include <stdio.h>
#include <stdlib.h>

int crearRed(struct network **resultado){
    struct network *res = malloc(sizeof(struct network));
    if(res == NULL){return -1;}

    res->capas = NULL;
    res->numeroCapas = 0;
    res->cache = NULL;
    *resultado = res;
    return 1;
}

int añadirCapa(struct layer *capa, struct network *red){
    if(red->capas == NULL){
        red->capas = malloc(sizeof(struct layer *));
        red->capas[0] = capa;
        red->cache = malloc(sizeof(struct cache *));
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
                temporal = realloc(red->cache, n * sizeof(struct cache *));
                if(temporal != NULL){
                    red->numeroCapas++;
                    return 1;
                }
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
        eliminarCacheLayer(&(*red)->cache[i]);
    }
    free((*red)->capas);
    free((*red)->cache);
    *red = NULL;
}

struct matrix *forwardRed(struct network *red, struct matrix *entrada){    
    struct matrix *copia;
    crearMatriz(&copia, 0, 0);
    copiarMatriz(entrada, copia);

    struct matrix *a;
    crearMatriz(&a, 0, 0);
    struct matrix *z;
    crearMatriz(&z, 0, 0);

    if(red->numeroCapas > 0 && red->capas[0]->nEntradas == entrada->fil && entrada->col == 1){
        for(int i=0; i<red->numeroCapas; i++){
            struct matrix *resultado = malloc(sizeof(struct matrix));
            crearMatriz(&resultado, red->capas[i]->nSalidas, 1); //en un futuro implementar batches (mas columnas) 
            red->cache[i] = forward(copia, red->capas[i], resultado);
            copiarMatriz(resultado, copia);
            eliminarMatriz(&resultado);
        }
        return copia;
    }
    return NULL;    

}


