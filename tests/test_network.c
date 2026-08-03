#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "layer.h"
#include "network.h"

int main(void)
{
    srand(time(NULL));

    printf("=====================================\n");
    printf("        TESTS MODULO NETWORK\n");
    printf("=====================================\n\n");

    /*====================================
      TEST 1: Crear red
    ====================================*/

    printf("===== TEST 1: Crear red =====\n");

    struct network *red;

    if(crearRed(&red))
        printf("Red creada correctamente.\n");
    else{
        printf("ERROR al crear la red.\n");
        return 1;
    }

    printf("Numero de capas: %d\n\n", red->numeroCapas);

    /*====================================
      TEST 2: Añadir primera capa
    ====================================*/

    printf("===== TEST 2: Añadir primera capa =====\n");

    struct matrix *pesos1;
    struct matrix *bias1;
    struct layer *L1;

    crearMatriz(&pesos1, 3, 4);
    crearMatriz(&bias1, 3, 1);

    inicializarRandom(pesos1, -1.0f, 1.0f);
    inicializarRandom(bias1, -1.0f, 1.0f);

    crearLayer(pesos1, bias1, RELU, &L1);

    if(añadirCapa(L1, red))
        printf("Primera capa añadida correctamente.\n");
    else
        printf("ERROR al añadir la primera capa.\n");

    printf("Numero de capas: %d\n\n", red->numeroCapas);

    /*====================================
      TEST 3: Añadir segunda capa compatible
    ====================================*/

    printf("===== TEST 3: Añadir segunda capa compatible =====\n");

    struct matrix *pesos2;
    struct matrix *bias2;
    struct layer *L2;

    crearMatriz(&pesos2, 2, 3);   // Compatible con la salida de L1 (3)
    crearMatriz(&bias2, 2, 1);

    inicializarRandom(pesos2, -1.0f, 1.0f);
    inicializarRandom(bias2, -1.0f, 1.0f);

    crearLayer(pesos2, bias2, SIGMOID, &L2);

    if(añadirCapa(L2, red))
        printf("Segunda capa añadida correctamente.\n");
    else
        printf("ERROR al añadir la segunda capa.\n");

    printf("Numero de capas: %d\n\n", red->numeroCapas);

    /*====================================
      TEST 4: Añadir capa incompatible
    ====================================*/

    printf("===== TEST 4: Capa incompatible =====\n");

    struct matrix *pesos3;
    struct matrix *bias3;
    struct layer *L3;

    crearMatriz(&pesos3, 5, 4);   // Espera 4 entradas, pero L2 produce 2
    crearMatriz(&bias3, 5, 1);

    inicializarRandom(pesos3, -1.0f, 1.0f);
    inicializarRandom(bias3, -1.0f, 1.0f);

    crearLayer(pesos3, bias3, TANH, &L3);

    if(!añadirCapa(L3, red))
        printf("Detectada correctamente la incompatibilidad.\n");
    else
        printf("ERROR: Se ha añadido una capa incompatible.\n");

    printf("Numero de capas (debe seguir siendo 2): %d\n\n", red->numeroCapas);

    /*====================================
      TEST 5: Mostrar arquitectura
    ====================================*/

    printf("===== TEST 5: Arquitectura =====\n");

    for(int i = 0; i < red->numeroCapas; i++)
    {
        printf("\nCapa %d\n", i + 1);
        printf("Entradas : %d\n", red->capas[i]->nEntradas);
        printf("Salidas  : %d\n", red->capas[i]->nSalidas);
    }

    /*====================================
      TEST 6: Eliminar red
    ====================================*/

    printf("\n===== TEST 6: Eliminar red =====\n");

    eliminarRed(&red);

    if(red == NULL)
        printf("Red eliminada correctamente.\n");
    else
        printf("ERROR al eliminar la red.\n");

    printf("\nTodos los tests finalizados.\n");

    return 0;
}