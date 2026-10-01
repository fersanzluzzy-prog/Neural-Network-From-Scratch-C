#ifndef NN_LOSS
#define NN_LOSS

#include "matrix.h"

float mse(struct matrix reales, struct matrix esperadas);

//resultados reales - resultados esperados
int mseDerivada(struct matrix reales, struct matrix esperadas, struct matrix *resultado);

#endif