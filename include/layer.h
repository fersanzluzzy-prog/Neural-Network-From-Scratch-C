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

};

int crearLayer(struct matrix *pesos, struct matrix *bias, activacion activacion, struct layer **resultado);

void eliminarLayer(struct layer **layer);

int forward(struct matrix *entradas, struct layer *layer, struct matrix *salidas);




#endif