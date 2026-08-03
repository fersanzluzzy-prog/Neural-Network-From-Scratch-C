#ifndef NN_ACTIVATION
#define NN_ACTIVATION

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int relu(struct matrix *entradas, struct matrix *salidas);

int sigmoide(struct matrix *entradas, struct matrix *salidas);

int taNh(struct matrix *entradas, struct matrix *salidas);

#endif