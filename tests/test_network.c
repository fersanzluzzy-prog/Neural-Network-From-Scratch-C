#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "layer.h"
#include "network.h"

int main(void)
{
    printf("=====================================\n");
    printf("        TESTS MODULO NETWORK\n");
    printf("=====================================\n\n");

    /*====================================
      TEST 1: Crear red
    ====================================*/

    printf("===== TEST 1: Crear red =====\n");

    struct network *red;

    if(crearRed(&red) == 1)
        printf("Red creada correctamente.\n");
    else{
        printf("ERROR al crear la red.\n");
        return 1;
    }

    printf("Numero de capas: %d\n", red->numeroCapas);

    /*====================================
      TEST 2: Añadir primera capa
    ====================================*/

    printf("\n===== TEST 2: Añadir primera capa =====\n");

    struct matrix *p1;
    struct matrix *b1;
    struct layer *L1;

    crearMatriz(&p1, 3, 2);
    crearMatriz(&b1, 3, 1);

    float datosP1[] = {
        1, 0,
        0, 1,
        1, 1
    };

    float datosB1[] = {
        0,
        0,
        0
    };

    inicializarMatriz(p1, datosP1);
    inicializarMatriz(b1, datosB1);

    crearLayer(p1, b1, RELU, &L1);

    if(añadirCapa(L1, red) == 1)
        printf("Primera capa añadida correctamente.\n");
    else
        printf("ERROR al añadir la primera capa.\n");

    printf("Numero de capas: %d\n", red->numeroCapas);

    /*====================================
      TEST 3: Añadir segunda capa
    ====================================*/

    printf("\n===== TEST 3: Añadir segunda capa =====\n");

    struct matrix *p2;
    struct matrix *b2;
    struct layer *L2;

    crearMatriz(&p2, 1, 3);
    crearMatriz(&b2, 1, 1);

    float datosP2[] = {
        1, 2, 3
    };

    float datosB2[] = {
        1
    };

    inicializarMatriz(p2, datosP2);
    inicializarMatriz(b2, datosB2);

    crearLayer(p2, b2, RELU, &L2);

    if(añadirCapa(L2, red) == 1)
        printf("Segunda capa añadida correctamente.\n");
    else
        printf("ERROR al añadir la segunda capa.\n");

    printf("Numero de capas: %d\n", red->numeroCapas);

    /*====================================
      TEST 4: Añadir capa incompatible
    ====================================*/

    printf("\n===== TEST 4: Capa incompatible =====\n");

    struct matrix *p3;
    struct matrix *b3;
    struct layer *L3;

    /*
        L2 produce 1 salida.

        L3 necesita 5 entradas:

        pesos = 4 x 5
        bias  = 4 x 1

        Por tanto:
        L2: 3 -> 1
        L3: 5 -> 4

        1 != 5 -> incompatible
    */

    crearMatriz(&p3, 4, 5);
    crearMatriz(&b3, 4, 1);

    float datosP3[] = {
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    };

    float datosB3[] = {
        0,
        0,
        0,
        0
    };

    inicializarMatriz(p3, datosP3);
    inicializarMatriz(b3, datosB3);

    crearLayer(p3, b3, RELU, &L3);

    int numeroCapasAntes = red->numeroCapas;

    if(añadirCapa(L3, red) == -1)
        printf("Incompatibilidad detectada correctamente.\n");
    else
        printf("ERROR: se ha añadido una capa incompatible.\n");

    if(red->numeroCapas == numeroCapasAntes)
        printf("El numero de capas no ha cambiado.\n");
    else
        printf("ERROR: el numero de capas ha cambiado.\n");

    /*====================================
      TEST 5: Forward de la red
    ====================================*/

    printf("\n===== TEST 5: Forward =====\n");

    struct matrix *entrada;

    crearMatriz(&entrada, 2, 1);

    float datosEntrada[] = {
        1,
        2
    };

    inicializarMatriz(entrada, datosEntrada);

    struct matrix *salida = forwardRed(red, entrada);

    if(salida != NULL)
    {
        printf("Forward correcto.\n");

        printf("\nEntrada:\n");
        imprimirMatriz(*entrada);

        printf("\nSalida obtenida:\n");
        imprimirMatriz(*salida);

        printf("\nResultado esperado:\n");
        printf("15.000000\n");
    }
    else
        printf("ERROR en forward.\n");

    /*====================================
      TEST 6: Forward con valores negativos
    ====================================*/

    printf("\n===== TEST 6: Forward con valores negativos =====\n");

    float datosEntradaNegativos[] = {
        -1,
        -2
    };

    inicializarMatriz(entrada, datosEntradaNegativos);

    salida = forwardRed(red, entrada);

    if(salida != NULL)
    {
        printf("Forward correcto.\n");

        printf("\nEntrada:\n");
        imprimirMatriz(*entrada);

        printf("\nSalida obtenida:\n");
        imprimirMatriz(*salida);

        printf("\nResultado esperado:\n");
        printf("1.000000\n");
    }
    else
        printf("ERROR en forward.\n");

    /*====================================
      TEST 7: Forward con red vacia
    ====================================*/

    printf("\n===== TEST 7: Forward con red vacia =====\n");

    struct network *redVacia;

    crearRed(&redVacia);

    struct matrix *resultadoVacio = forwardRed(redVacia, entrada);

    if(resultadoVacio == NULL)
        printf("Error detectado correctamente.\n");
    else
        printf("ERROR: se ha realizado forward sobre una red vacia.\n");

    eliminarRed(&redVacia);

    /*====================================
      TEST 8: Eliminar red
    ====================================*/

    printf("\n===== TEST 8: Eliminar red =====\n");

    eliminarRed(&red);

    if(red == NULL)
        printf("Red eliminada correctamente.\n");
    else
        printf("ERROR al eliminar la red.\n");

    printf("\n=====================================\n");
    printf("       TODOS LOS TESTS FINALIZADOS\n");
    printf("=====================================\n");

    return 0;
}