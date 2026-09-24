#include "network.h"
#include "loss.h"
#include <stdio.h>
#include <stdlib.h>

int crearRed(struct network **resultado){
    struct network *res = malloc(sizeof(struct network));
    if(res == NULL){return -1;}

    res->capas = NULL;
    res->numeroCapas = 0;
    res->cache = NULL;
    *resultado = res;
    res->entradas = NULL;
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
    free((*red)->entradas);
    *red = NULL;
}

struct matrix *forwardRed(struct network *red, struct matrix *entrada){    
    
    
    struct matrix *copia;
    crearMatriz(&copia, 0, 0);
    copiarMatriz(entrada, copia);
    red->entradas = malloc(sizeof(struct matrix));
    copiarMatriz(entrada, red->entradas);

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

int backpropRed(struct network *red, struct matrix resReales, struct matrix resCorrectos){
    if(resCorrectos.col == resReales.col && resCorrectos.fil == resReales.fil){
        for(int i = red->numeroCapas - 1; i>=0; i++){
            red->capas[i]->gradientes = malloc(sizeof(struct gradientesLayer));
            red->capas[i]->gradientes->dL_db = malloc(sizeof(struct matrix));
            red->capas[i]->gradientes->dL_dw = malloc(sizeof(struct matrix));
            red->capas[i]->gradientes->dL_dz = malloc(sizeof(struct matrix));

            //da_dz
            struct matrix *da_dz = malloc(sizeof(struct matrix));
            crearMatriz(&da_dz, resCorrectos.fil, resCorrectos.col);
            activacion actv = red->capas[i]->activacion;
            if(actv == RELU){
                reluDerivada(red->cache[i]->z, da_dz);
            }
            else if(actv == SIGMOID){
                sigmoideDerivada(red->cache[i]->z, da_dz);
            }
            else if(actv == TANH){
                taNhDerivada(red->cache[i]->z, da_dz);
            }

            //gradientes
            struct matrix *dz_dw = malloc(sizeof(struct matrix));
            if(i>0){
                copiarMatriz(red->cache[i-1]->a, dz_dw); //batches en un futuro
            }
            else{copiarMatriz(red->entradas, dz_dw);}

            struct matrix *res = malloc(sizeof(struct matrix));
            crearMatriz(&res, resReales.fil, 1); //batches en un futuro

            if(i == red->numeroCapas - 1){
                //dL_da y dL_dz
                struct matrix *dL_da = malloc(sizeof(struct matrix));
                crearMatriz(&dL_da, resCorrectos.fil, resCorrectos.col);
                mseDerivada(resReales, resCorrectos, dL_da);
                multiplicacionMatricial(dL_da, da_dz, res);
            }  
            else{res = red->capas[i-1]->gradientes->dL_dz;}
            transponerMatriz(res,red->capas[i]->gradientes->dL_dz);

            // dL/dw
            struct matrix *dL_dw = malloc(sizeof(struct matrix));
            crearMatriz(&dL_dw, red->capas[i]->nEntradas, red->capas[i]->nSalidas);
            if(i!=0){
                struct matrix *aux = malloc(sizeof(struct matrix));
                crearMatriz(&aux, red->cache[i-1]->a->fil, red->cache[i-1]->a->col);
                transponerMatriz(red->cache[i-1]->a, aux);
                multiplicacionMatricial(res, aux, dL_dw);
                free(aux);
            }
            else{
                struct matrix *aux = malloc(sizeof(struct matrix));
                crearMatriz(&aux, red->cache[i-1]->a->fil, red->cache[i-1]->a->col);
                transponerMatriz(red->cache[i-1]->a, aux);
                free(aux);
                multiplicacionMatricial(res, red->entradas, dL_dw);
            }
            *red->capas[i]->gradientes->dL_dw = *dL_dw;
            // dL/db
            struct matrix *dL_db = malloc(sizeof(struct matrix));
            crearMatriz(&dL_db, red->capas[i]->bias->fil, red->capas[i]->bias->col);
            struct matrix *unos = malloc(sizeof(struct matrix));
            crearConNumero(&unos, red->capas[i]->bias->fil, red->capas[i]->bias->col, 1.0);
            multiplicacionMatricial(res, unos, dL_db);
            *red->capas[i]->gradientes->dL_db = *dL_db;

            free(da_dz);
            free(dz_dw);
            free(dL_dw);
            free(dL_db);
        }

        return 1;
    }
    return -1;
}










