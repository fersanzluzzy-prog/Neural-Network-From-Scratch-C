#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "layer.h"

int main(void)
{
    srand(time(NULL));

    printf("=====================================\n");
    printf("        TESTS MODULO LAYER\n");
    printf("=====================================\n\n");

    /*====================================
      TEST 1: Crear Layer
    ====================================*/

    printf("===== TEST 1: Crear Layer =====\n");

    struct matrix *pesos;
    struct matrix *bias;
    struct layer *L;

    crearMatriz(&pesos, 2, 3);
    crearMatriz(&bias, 2, 1);

    inicializarRandom(pesos, -1.0f, 1.0f);
    inicializarRandom(bias, -1.0f, 1.0f);

    if (crearLayer(pesos, bias, &L))
        printf("Layer creada correctamente.\n");
    else
    {
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
      TEST 2: Forward
    ====================================*/

    printf("\n===== TEST 2: Forward =====\n");

    float datosPesos[] = {
        1, 2, 3,
        4, 5, 6
    };

    float datosBias[] = {
        10,
        20
    };

    inicializarMatriz(L->pesos, datosPesos);
    inicializarMatriz(L->bias, datosBias);

    struct matrix *entrada;
    crearMatriz(&entrada, 3, 1);

    float datosEntrada[] = {
        7,
        8,
        9
    };

    inicializarMatriz(entrada, datosEntrada);

    struct matrix *salida;
    crearMatriz(&salida, 2, 1);

    if (forward(entrada, L, salida))
        printf("Forward correcto.\n");
    else
        printf("ERROR en forward.\n");

    printf("\nEntrada:\n");
    imprimirMatriz(*entrada);

    printf("\nSalida obtenida:\n");
    imprimirMatriz(*salida);

    printf("\nResultado esperado:\n");
    printf("60.000000\n");
    printf("162.000000\n");

    /*====================================
      TEST 3: Eliminar Layer
    ====================================*/

    printf("\n===== TEST 3: Eliminar Layer =====\n");

    eliminarLayer(&L);

    printf("Layer = %p\n", (void *)L);

    return 0;
}