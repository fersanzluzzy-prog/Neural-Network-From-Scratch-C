#include <stdio.h>

#include "matrix.h"
#include "loss.h"

int main(void)
{
    printf("=====================================\n");
    printf("        TESTS MODULO LOSS\n");
    printf("=====================================\n\n");

    /*====================================
      TEST 1: Error cero
    ====================================*/

    printf("===== TEST 1: Prediccion perfecta =====\n");

    struct matrix *reales1;
    struct matrix *esperadas1;

    crearMatriz(&reales1, 3, 1);
    crearMatriz(&esperadas1, 3, 1);

    float datosReales1[] = {
        1,
        0,
        0
    };

    float datosEsperadas1[] = {
        1,
        0,
        0
    };

    inicializarMatriz(reales1, datosReales1);
    inicializarMatriz(esperadas1, datosEsperadas1);

    float resultado = mse(*reales1, *esperadas1);

    printf("MSE: %.6f\n", resultado);
    printf("Esperado: 0.000000\n");

    if(resultado == 0.0f)
        printf("CORRECTO\n");
    else
        printf("ERROR\n");


    /*====================================
      TEST 2: Una salida
    ====================================*/

    printf("\n===== TEST 2: Una salida =====\n");

    struct matrix *reales2;
    struct matrix *esperadas2;

    crearMatriz(&reales2, 1, 1);
    crearMatriz(&esperadas2, 1, 1);

    float datosReales2[] = {
        0.5f
    };

    float datosEsperadas2[] = {
        1.0f
    };

    inicializarMatriz(reales2, datosReales2);
    inicializarMatriz(esperadas2, datosEsperadas2);

    resultado = mse(*reales2, *esperadas2);

    printf("Real:      0.500000\n");
    printf("Esperada:  1.000000\n");
    printf("MSE: %.6f\n", resultado);
    printf("Esperado: 0.125000\n");

    if(resultado == 0.125f)
        printf("CORRECTO\n");
    else
        printf("ERROR\n");


    /*====================================
      TEST 3: Varias salidas
    ====================================*/

    printf("\n===== TEST 3: Varias salidas =====\n");

    struct matrix *reales3;
    struct matrix *esperadas3;

    crearMatriz(&reales3, 3, 1);
    crearMatriz(&esperadas3, 3, 1);

    float datosReales3[] = {
        0.2f,
        0.7f,
        0.1f
    };

    float datosEsperadas3[] = {
        0.0f,
        1.0f,
        0.0f
    };

    inicializarMatriz(reales3, datosReales3);
    inicializarMatriz(esperadas3, datosEsperadas3);

    resultado = mse(*reales3, *esperadas3);

    printf("MSE: %.6f\n", resultado);
    printf("Esperado: 0.023333\n");

    if(resultado > 0.023333f && resultado < 0.023333f)
        printf("CORRECTO\n");
    else
        printf("ERROR\n");


    /*====================================
      TEST 4: Errores negativos
    ====================================*/

    printf("\n===== TEST 4: Valores negativos =====\n");

    struct matrix *reales4;
    struct matrix *esperadas4;

    crearMatriz(&reales4, 2, 1);
    crearMatriz(&esperadas4, 2, 1);

    float datosReales4[] = {
        -1.0f,
        -2.0f
    };

    float datosEsperadas4[] = {
        1.0f,
        2.0f
    };

    inicializarMatriz(reales4, datosReales4);
    inicializarMatriz(esperadas4, datosEsperadas4);

    resultado = mse(*reales4, *esperadas4);

    printf("MSE: %.6f\n", resultado);
    printf("Esperado: 5.000000\n");

    if(resultado == 5.0f)
        printf("CORRECTO\n");
    else
        printf("ERROR\n");


    /*====================================
      Liberar memoria
    ====================================*/

    eliminarMatriz(&reales1);
    eliminarMatriz(&esperadas1);

    eliminarMatriz(&reales2);
    eliminarMatriz(&esperadas2);

    eliminarMatriz(&reales3);
    eliminarMatriz(&esperadas3);

    eliminarMatriz(&reales4);
    eliminarMatriz(&esperadas4);

    printf("\n=====================================\n");
    printf("       TODOS LOS TESTS FINALIZADOS\n");
    printf("=====================================\n");

    return 0;
}