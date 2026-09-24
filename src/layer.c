#include "layer.h"
#include "loss.h"
#include <stdio.h>
#include <stdlib.h>

//si devuelven int, 1 si va todo bien, -1 si ocurre algun error

int crearLayer(struct matrix *pesos, struct matrix *bias, activacion activacion, struct layer **resultado){
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
        (*resultado)->activacion = activacion;
        (*resultado)->gradientes = NULL;
        return 1;
    }
    return -1;
}

void eliminarLayer(struct layer **layer){
    eliminarMatriz(&(*layer)->pesos);
    eliminarMatriz(&(*layer)->bias);
    free((*layer)->gradientes);
    free(*layer);
    *layer = NULL;
}

//retorna un puntero a una struct cache con a y con z, NULL si algo sale mal
struct cache *forward(struct matrix *entradas, struct layer *layer, struct matrix *salidas){
    if (entradas->fil == layer->nEntradas && salidas->fil == layer->nSalidas){
        struct matrix *copia = salidas; 
        struct cache *c = inicializarCache();

        multiplicacionMatricial(layer->pesos, entradas, salidas);
        suma(*copia, *layer->bias, salidas);
        guardarCacheLayerZ(c, *salidas);
        
        activacion actv = layer->activacion;
        if(actv == RELU){
            *copia = *salidas; 
            relu(copia, salidas);
        }
        else if(actv == SIGMOID){
            *copia = *salidas; 
            sigmoide(copia, salidas);
        }
        else if(actv == TANH){
            *copia = *salidas; 
            taNh(copia, salidas);
        }

        guardarCacheLayerA(c, *salidas);

        return c;
    }
    else{return NULL;}
}

struct cache *inicializarCache(){
    struct cache *cache = malloc(sizeof(struct cache));
    cache->a = NULL;
    cache->z = NULL;
    return cache;
}

void guardarCacheLayerA(struct cache *cache, struct matrix a1){
    cache->a = malloc(sizeof(struct matrix));
    struct matrix *m;
    crearMatriz(&m, a1.fil, a1.col);
    copiarMatriz(&a1, m);
    cache->a = m;
}

void guardarCacheLayerZ(struct cache *cache, struct matrix z1){
    cache->z = malloc(sizeof(struct matrix));
    struct matrix *m;
    crearMatriz(&m, z1.fil, z1.col);
    copiarMatriz(&z1, m);
    cache->z = m;
}

int eliminarCacheLayer(struct cache **c){
    free((*c)->a);
    free((*c)->z);
    free(*c);
    *c = NULL;
    return 1;
}
