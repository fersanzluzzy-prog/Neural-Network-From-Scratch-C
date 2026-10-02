#include "network.h"
#include "loss.h"
#include <stdio.h>
#include <stdlib.h>

int crearRed(struct network **resultado){

    *resultado = malloc(sizeof(struct network));
    (*resultado)->capas = malloc(sizeof(struct layer *));
    (*resultado)->numeroCapas = 0;
    (*resultado)->cache = malloc(sizeof(struct cache *));
    (*resultado)->entradas = malloc(sizeof(struct matrix));
    return 0;
}

int añadirCapa(struct layer *capa, struct network *red){
    int n = red->numeroCapas;
    int nEntradasCapaNueva = capa->nEntradas;
    int nSalidasUltimaCapa;
    if(n == 0){
        red->entradas->col = capa->bias->col;
        red->entradas->fil = capa->bias->fil;
        nSalidasUltimaCapa = nEntradasCapaNueva;
    }
    else {nSalidasUltimaCapa = red->capas[n-1]->nSalidas;}
    if(nEntradasCapaNueva == nSalidasUltimaCapa){
        struct layer **temporal1 = realloc(red->capas, (n+1) * sizeof(struct layer *));
        if(temporal1 != NULL){
            red->capas = temporal1;
            red->capas[n] = capa;
            struct cache **temporal2 = realloc(red->cache, (n+1) * sizeof(struct cache *));
            if(temporal2 != NULL){
                red->cache = temporal2;
                red->numeroCapas++;
                red->cache[n] = NULL;
                return 0;
            }
        }
    }
    return 1;
}

void eliminarRed(struct network **red){

    if (red == NULL || *red == NULL)
        return;

    int n = (*red)->numeroCapas;

    for (int i = 0; i < n; i++) {
        eliminarLayer(&(*red)->capas[i]);
        eliminarCacheLayer(&(*red)->cache[i]);
    }

    free((*red)->capas);
    free((*red)->cache);
    free((*red)->entradas);

    free(*red);
    *red = NULL;
}

struct matrix *forwardRed(struct network *red, struct matrix *entrada){    
    
    
    struct matrix *copia;
    crearMatriz(&copia, 0, 0);
    copiarMatriz(entrada, copia);
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
        for(int i = red->numeroCapas - 1; i>=0; i--){

            struct matrix *dL_da = malloc(sizeof(struct matrix));
            struct matrix *da_dz = malloc(sizeof(struct matrix));
            struct matrix *dL_dx = malloc(sizeof(struct matrix));
            struct matrix *dL_dW = malloc(sizeof(struct matrix));
            struct matrix *x = malloc(sizeof(struct matrix));
            struct matrix *dL_dz = malloc(sizeof(struct matrix));

            struct matrix *wT = malloc(sizeof(struct matrix));
            struct matrix *xT = malloc(sizeof(struct matrix));


            red->capas[i]->gradientes = malloc(sizeof(struct gradientesLayer));
            red->capas[i]->gradientes->dL_db = malloc(sizeof(struct matrix));
            red->capas[i]->gradientes->dL_dw = malloc(sizeof(struct matrix));
            red->capas[i]->gradientes->dL_dx = malloc(sizeof(struct matrix));    
            
            //caso: solo 1 capa (puto test de chatgpt)
            if(red->numeroCapas == 1){
                //dL_da
                crearMatriz(&dL_da, resReales.fil, resReales.col);
                mseDerivada(resReales, resCorrectos, dL_da);

                //dL_dx
                crearMatriz(&dL_dx, red->entradas->fil, red->entradas->col);

                //x
                crearMatriz(&x, red->entradas->fil, red->entradas->col);
                copiarMatriz(red->entradas, x);
            }            
            //ultima capa
            else if(i == red->numeroCapas - 1){
                //dL_da
                crearMatriz(&dL_da, resReales.fil, resReales.col);
                mseDerivada(resReales, resCorrectos, dL_da);

                //dL_dx
                crearMatriz(&dL_dx, red->cache[i-1]->a->fil, red->cache[i-1]->a->col);

                //x
                crearMatriz(&x, red->cache[i-1]->a->fil, red->cache[i-1]->a->col);
                copiarMatriz(red->cache[i-1]->a, x);

            }
            //primera capa
            else if(i == 0){

                //dL_da
                crearMatriz(&dL_da, red->capas[i+1]->gradientes->dL_dx->fil, red->capas[i+1]->gradientes->dL_dx->fil);
                copiarMatriz(red->capas[i+1]->gradientes->dL_dx, dL_da);

                //dL_dx
                crearMatriz(&dL_dx, red->entradas->fil, red->entradas->col);

                //x
                crearMatriz(&x, red->entradas->fil, red->entradas->col);
                copiarMatriz(red->entradas, x);

            }
            //capas intermedias
            else{

                //dL_da
                crearMatriz(&dL_da, red->capas[i+1]->gradientes->dL_dx->fil, red->capas[i+1]->gradientes->dL_dx->fil);
                copiarMatriz(red->capas[i+1]->gradientes->dL_dx, dL_da);
                
                //dL_dx
                crearMatriz(&dL_dx, red->cache[i-1]->a->fil, red->cache[i-1]->a->col);
            
                //x
                crearMatriz(&x, red->cache[i-1]->a->fil, red->cache[i-1]->a->col);
                copiarMatriz(red->cache[i-1]->a, x);

            }
            //da_dz
            crearMatriz(&da_dz, dL_da->fil, dL_da->col);

            activacion actv = red->capas[i]->activacion;
            if(actv == RELU){
                reluDerivada(red->cache[i]->z, da_dz);
            }
            else if(actv == SIGMOID){
                sigmoideDerivada(red->cache[i]->a, da_dz);
            }
            else if(actv == TANH){
                taNhDerivada(red->cache[i]->a, da_dz);
            }

            //dL_dz = dl_da · da_dz
            crearMatriz(&dL_dz, dL_da->fil, dL_da->col);
            multiplicacionElemPorElem(dL_da, da_dz, dL_dz);
            //dL_dw = dL_dz x xT
            crearMatriz(&dL_dW, red->capas[i]->pesos->fil, red->capas[i]->pesos->col);
            crearMatriz(&xT, x->col, x->fil);
            transponerMatriz(x, xT);
            multiplicacionMatricial(dL_dz, xT, dL_dW);
            //dL_dx = wT x dL_dz
            crearMatriz(&wT, red->capas[i]->pesos->col, red->capas[i]->pesos->fil);
            transponerMatriz(red->capas[i]->pesos, wT);
            multiplicacionMatricial(wT, dL_dz, dL_dx);
            //dL_db = dL_dz
            //batches, lo voy a obviar hasta añadirlos

            //guardar gradientes
            copiarMatriz(dL_dz, red->capas[i]->gradientes->dL_db); //batches
            copiarMatriz(dL_dW, red->capas[i]->gradientes->dL_dw);
            copiarMatriz(dL_dx, red->capas[i]->gradientes->dL_dx);
            eliminarMatriz(&dL_da);
            eliminarMatriz(&da_dz);
            eliminarMatriz(&dL_dx);
            eliminarMatriz(&dL_dW);
            eliminarMatriz(&dL_dz);
            eliminarMatriz(&wT);
            eliminarMatriz(&xT);
        }
        return 0;
    }
    return 1;
}

int descensoDeGradiente(struct network *red, float factorAprendizaje){
    for(int i=0; i<red->numeroCapas; i++){
        struct matrix *pesos = red->capas[i]->pesos;
        struct matrix *bias = red->capas[i]->bias;

        struct matrix *dw = red->capas[i]->gradientes->dL_dw;
        struct matrix *db = red->capas[i]->gradientes->dL_db;

        for(int k=0; k<pesos->fil; k++){
            bias->datos[k] = bias->datos[k] - (factorAprendizaje * db->datos[k]);
            for(int j=0; j<pesos->col; j++){
                int indice = k*pesos->col + j;
                pesos->datos[indice] = pesos->datos[indice] - (factorAprendizaje * dw->datos[indice]);
            }
        }
    }
    return 0;
}









