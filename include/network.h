#ifndef NN_NETWORK
#define NN_NETWORK

#include "layer.h"
#include "matrix.h"

struct network{
    struct layer **capas;
    int numeroCapas;
};

int crearRed(struct network **resultado);

int añadirCapa(struct layer *capa, struct network *red);

void eliminarRed(struct network **red);

struct matrix *forwardRed(struct network *red, struct matrix *entrada);
#endif