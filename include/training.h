#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "loss.h"

//hay que poner el número de entradas que le pasas, mismo numero de entradas reales que de esperadas y en mismo orden
int training(struct network *red, struct matrix **entradas, struct matrix **resultadosEsperados, float factorAprendizaje, int nEntradas, int epocas);
