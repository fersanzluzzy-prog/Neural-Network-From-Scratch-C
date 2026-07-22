#include <stdio.h>

#include "matrix.h"
#include "activation.h"

int main(void)
{
    printf("=====================================\n");
    printf("      TESTS MODULO ACTIVATION\n");
    printf("=====================================\n");

    struct matrix *entrada;
    struct matrix *salida;

    crearMatriz(&entrada, 3, 3);
    crearMatriz(&salida, 3, 3);

    float datos[] = {
        -3.0f, -2.0f, -1.0f,
         0.0f,  1.0f,  2.0f,
         3.0f,  4.0f,  5.0f
    };

    inicializarMatriz(entrada, datos);

    printf("\n===== TEST 1: Entrada =====\n");
    imprimirMatriz(*entrada);

    if(relu(entrada, salida))
        printf("\nReLU ejecutada correctamente.\n");
    else
        printf("\nERROR EN RELU\n");

    printf("\n===== TEST 2: Salida =====\n");
    imprimirMatriz(*salida);

    printf("\nEsperado:\n");
    printf("0 0 0\n");
    printf("0 1 2\n");
    printf("3 4 5\n");

    /* Comprobar que la entrada no cambia */

    printf("\n===== TEST 3: La entrada permanece igual =====\n");
    imprimirMatriz(*entrada);

    eliminarMatriz(&entrada);
    eliminarMatriz(&salida);

    printf("\nTodos los tests finalizados.\n");

    return 0;
}