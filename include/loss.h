#ifndef NN_LOSS
#define NN_LOSS

float mse(struct matrix reales, struct matrix esperadas);

struct matrix *mseDerivada(struct matrix reales, struct matrix esperadas);

#endif