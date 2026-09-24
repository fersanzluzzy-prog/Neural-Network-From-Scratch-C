#ifndef NN_LOSS
#define NN_LOSS

float mse(struct matrix reales, struct matrix esperadas);

int mseDerivada(struct matrix reales, struct matrix esperadas, struct matrix *resultado);

#endif