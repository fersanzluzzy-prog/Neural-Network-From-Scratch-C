#ifndef NN_ACTIVATION
#define NN_ACTIVATION

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int relu(struct matrix *entradas, struct matrix *salidas);
int reluDerivada(struct matrix *entradas, struct matrix *salidas);

int sigmoide(struct matrix *entradas, struct matrix *salidas);
//hay que pasarle las entradas ya evaluadas por sigmoide()
int sigmoideDerivada(struct matrix *entradasSigmoide, struct matrix *salidas);

int taNh(struct matrix *entradas, struct matrix *salidas);
//hay que pasarle las entradas ya evaluadas por taNh()
int taNhDerivada(struct matrix *entradas, struct matrix *salidas);

#endif