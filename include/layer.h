#ifndef NN_LAYER
#define NN_LAYER

#include "matrix.h"

struct layer{
    struct matrix *pesos;
    struct matrix *bias;
    int nEntradas;
    int nSalidas;
};

int crearLayer(struct matrix *pesos, struct matrix *bias, struct layer **resultado);

void eliminarLayer(struct layer **layer);

int forward(struct matrix *entradas, struct layer *layer, struct matrix *salidas);




#endif