#ifndef NN_NETWORK
#define NN_NETWORK

#include "layer.h"
#include "matrix.h"

struct network{
    struct layer **capas;
    struct cache **cache;
    int numeroCapas;
    struct matrix *entradas;
};

int crearRed(struct network **resultado);

int añadirCapa(struct layer *capa, struct network *red);

void eliminarRed(struct network **red);

struct matrix *forwardRed(struct network *red, struct matrix *entrada);

int backpropRed(struct network *red, struct matrix resReales, struct matrix resCorrectos);

#endif