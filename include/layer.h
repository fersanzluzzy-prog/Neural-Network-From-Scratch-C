#ifndef NN_LAYER
#define NN_LAYER

#include "matrix.h"
#include "activation.h"

typedef enum{
    RELU,
    SIGMOID,
    TANH    
} activacion;

struct layer{
    struct matrix *pesos;
    struct matrix *bias;
    int nEntradas;
    int nSalidas;
    activacion activacion;
    struct gradientesLayer *gradientes;
};

struct cache{
    struct matrix *a;
    struct matrix *z;
};

struct gradientesLayer{
    struct matrix *dL_dw;
    struct matrix *dL_db;
    struct matrix *dL_dx;
};

int crearLayer(struct matrix *pesos, struct matrix *bias, activacion activacion, struct layer **resultado);

void eliminarLayer(struct layer **layer);

struct cache *forward(struct matrix *entradas, struct layer *layer, struct matrix *salidas);

struct cache *inicializarCache();

void guardarCacheLayerA(struct cache *cache, struct matrix a1);

void guardarCacheLayerZ(struct cache *cache, struct matrix z1);

int eliminarCacheLayer(struct cache **c);


#endif