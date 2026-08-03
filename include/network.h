#ifndef NN_NETWORK
#define NN_NETWORK

#include "layer.h"

struct network{
    struct layer **capas;
    int numeroCapas;
};

int crearRed(struct network **resultado);
int añadirCapa(struct layer *capa, struct network *red);
void eliminarRed(struct network **red);

#endif