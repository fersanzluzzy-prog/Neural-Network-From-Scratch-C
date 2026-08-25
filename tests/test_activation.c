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

    

    printf("\nTodos los tests finalizados.\n");

    printf("\n===== TEST 4: Sigmoid =====\n");

    float datosSigmoid[] = {
        -2.0f, -1.0f, 0.0f,
        1.0f,  2.0f, 3.0f,
        -3.0f,  4.0f, 5.0f
    };

    inicializarMatriz(entrada, datosSigmoid);

    printf("\nEntrada:\n");
    imprimirMatriz(*entrada);

    if(sigmoide(entrada, salida))
        printf("\nSigmoid ejecutada correctamente.\n");
    else
        printf("\nERROR EN SIGMOID\n");

    printf("\nSalida obtenida:\n");
    imprimirMatriz(*salida);

    printf("\nValores aproximados esperados:\n");
    printf("0.119203 0.268941 0.500000\n");
    printf("0.731059 0.880797 0.952574\n");
    printf("0.047426 0.982014 0.993307\n");

    /*====================================
    TEST 4: Tanh
    ====================================*/

    printf("\n===== TEST 4: Tanh =====\n");

    float datosTanh[] = {
        -2.0f, -1.0f, 0.0f,
        1.0f,  2.0f, 3.0f,
        -3.0f,  4.0f, 5.0f
    };

    inicializarMatriz(entrada, datosTanh);

    printf("\nEntrada:\n");
    imprimirMatriz(*entrada);

    if(taNh(entrada, salida))
        printf("\nTanh ejecutada correctamente.\n");
    else
        printf("\nERROR EN TANH\n");

    printf("\nSalida obtenida:\n");
    imprimirMatriz(*salida);

    printf("\nValores aproximados esperados:\n");
    printf("-0.964028 -0.761594  0.000000\n");
    printf(" 0.761594  0.964028  0.995055\n");
    printf("-0.995055  0.999329  0.999909\n");

    eliminarMatriz(&entrada);
    eliminarMatriz(&salida);

    /*====================================
    TESTS DERIVADAS
    ====================================*/

    printf("\n=====================================\n");
    printf("       TESTS DE DERIVADAS\n");
    printf("=====================================\n");

    struct matrix *entradaDerivada;
    struct matrix *salidaDerivada;
    struct matrix *auxiliarDerivada;

    crearMatriz(&entradaDerivada, 3, 3);
    crearMatriz(&salidaDerivada, 3, 3);
    crearMatriz(&auxiliarDerivada, 3, 3);


    float datosDerivadas[] = {
        -2.0f, -1.0f, 0.0f,
         1.0f,  2.0f, 3.0f,
        -3.0f,  4.0f, 5.0f
    };

    inicializarMatriz(entradaDerivada, datosDerivadas);

    /*====================================
    TEST 5: Derivada ReLU
    ====================================*/

    printf("\n===== TEST 5: Derivada ReLU =====\n");

    printf("\nEntrada:\n");
    imprimirMatriz(*entradaDerivada);

    if(reluDerivada(entradaDerivada, salidaDerivada))
        printf("\nDerivada ReLU ejecutada correctamente.\n");
    else
        printf("\nERROR EN DERIVADA RELU\n");

    printf("\nSalida obtenida:\n");
    imprimirMatriz(*salidaDerivada);

    printf("\nValores esperados:\n");
    printf("0 0 0\n");
    printf("1 1 1\n");
    printf("0 1 1\n");

    /*====================================
    TEST 6: Derivada Sigmoid
    ====================================*/

    printf("\n===== TEST 6: Derivada Sigmoid =====\n");

    printf("\nEntrada:\n");
    imprimirMatriz(*entradaDerivada);

    sigmoide(entradaDerivada, auxiliarDerivada);

    if(sigmoideDerivada(auxiliarDerivada, salidaDerivada))
        printf("\nDerivada Sigmoid ejecutada correctamente.\n");
    else
        printf("\nERROR EN DERIVADA SIGMOID\n");

    printf("\nSalida obtenida:\n");
    imprimirMatriz(*salidaDerivada);

    printf("\nValores aproximados esperados:\n");
    printf("0.104994 0.196612 0.250000\n");
    printf("0.196612 0.104994 0.045177\n");
    printf("0.045177 0.017663 0.006648\n");

    /*====================================
    TEST 7: Derivada Tanh
    ====================================*/

    printf("\n===== TEST 7: Derivada Tanh =====\n");

    printf("\nEntrada:\n");
    imprimirMatriz(*entradaDerivada);
    taNh(entradaDerivada, auxiliarDerivada);

    if(taNhDerivada(auxiliarDerivada, salidaDerivada))
        printf("\nDerivada Tanh ejecutada correctamente.\n");
    else
        printf("\nERROR EN DERIVADA TANH\n");

    printf("\nSalida obtenida:\n");
    imprimirMatriz(*salidaDerivada);

    printf("\nValores aproximados esperados:\n");
    printf("0.070651 0.419974 1.000000\n");
    printf("0.419974 0.070651 0.009866\n");
    printf("0.009866 0.001341 0.000182\n");

    eliminarMatriz(&entradaDerivada);
    eliminarMatriz(&salidaDerivada);

    return 0;
}