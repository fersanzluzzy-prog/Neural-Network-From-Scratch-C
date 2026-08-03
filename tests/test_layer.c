#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "layer.h"

int main(void)
{
    srand(time(NULL));

    printf("=====================================\n");
    printf("         TESTS MODULO LAYER\n");
    printf("=====================================\n");

    /*====================================
      TEST 1: Crear Layer
    ====================================*/

    printf("\n===== TEST 1: Crear Layer =====\n");

    struct matrix *pesos;
    struct matrix *bias;
    struct layer *L;

    crearMatriz(&pesos, 2, 3);
    crearMatriz(&bias, 2, 1);

    float datosPesos[] = {
        0.5f, -1.0f, 2.0f,
        1.0f,  0.0f,-0.5f
    };

    float datosBias[] = {
        0.1f,
       -0.2f
    };

    inicializarMatriz(pesos, datosPesos);
    inicializarMatriz(bias, datosBias);

    if(crearLayer(pesos, bias, RELU, &L))
        printf("Layer creada correctamente.\n");
    else{
        printf("ERROR al crear la layer.\n");
        return 1;
    }

    printf("\nPesos:\n");
    imprimirMatriz(*L->pesos);

    printf("\nBias:\n");
    imprimirMatriz(*L->bias);

    printf("\nEntradas: %d\n", L->nEntradas);
    printf("Salidas : %d\n", L->nSalidas);


    /*====================================
      TEST 3: Forward RELU
    ====================================*/

    printf("\n===== TEST 3: Forward RELU =====\n");

    struct matrix *entrada;
    struct matrix *salida;
    struct matrix *prueba;

    crearMatriz(&entrada,3,1);
    crearMatriz(&salida,2,1);
    crearMatriz(&prueba, 2, 1);

    float datosEntrada[] = {
        1.0f,
        2.0f,
       -1.0f
    };

    inicializarMatriz(entrada,datosEntrada);

    L->activacion = RELU;

    if(forward(entrada,L,salida))
        printf("Forward correcto.\n");
    else
        printf("ERROR en forward.\n");

    printf("\nSalida:\n");
    imprimirMatriz(*salida);

    printf("\nEsperado:\n");
    printf("0.000000\n");
    printf("1.300000\n");

    /*====================================
      TEST 4: Forward SIGMOID
    ====================================*/

    printf("\n===== TEST 4: Forward SIGMOID =====\n");

    L->activacion = SIGMOID;

    forward(entrada,L,salida);

    printf("\nSalida:\n");
    imprimirMatriz(*salida);

    printf("\nEsperado (aprox.):\n");
    printf("0.032295\n");
    printf("0.785835\n");

    /*====================================
      TEST 5: Forward TANH
    ====================================*/

    printf("\n===== TEST 5: Forward TANH =====\n");

    L->activacion = TANH;

    forward(entrada,L,salida);

    printf("\nSalida:\n");
    imprimirMatriz(*salida);

    printf("\nEsperado (aprox.):\n");
    printf("-0.997775\n");
    printf("0.861723\n");

    /*====================================
      TEST 6: Eliminar Layer
    ====================================*/

    printf("\n===== TEST 6: Eliminar Layer =====\n");

    eliminarLayer(&L);

    printf("Layer = %p\n",(void*)L);

    return 0;
}