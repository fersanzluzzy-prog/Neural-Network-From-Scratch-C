#include <stdio.h>

#include "matrix.h"
#include "layer.h"

int main(void)
{
    printf("=====================================\n");
    printf("        TESTS CACHE DE LAYER\n");
    printf("=====================================\n\n");

    /*====================================
      TEST 1: Inicializar cache
    ====================================*/

    printf("===== TEST 1: Inicializar cache =====\n");

    struct cache *cache = inicializarCache();

    if(cache != NULL)
        printf("Cache creada correctamente.\n");
    else{
        printf("ERROR al crear la cache.\n");
        return 1;
    }

    /*====================================
      TEST 2: Guardar A y Z
    ====================================*/

    printf("\n===== TEST 2: Guardar A y Z =====\n");

    struct matrix *a1;
    struct matrix *z1;

    crearMatriz(&a1, 3, 1);
    crearMatriz(&z1, 2, 1);

    float datosA[] = {
        1,
        2,
        3
    };

    float datosZ[] = {
        10,
        20
    };

    inicializarMatriz(a1, datosA);
    inicializarMatriz(z1, datosZ);

    guardarCacheLayerA(cache, *a1);
    guardarCacheLayerZ(cache, *z1);

    printf("A guardada:\n");
    imprimirMatriz(*cache->a);

    printf("\nZ guardada:\n");
    imprimirMatriz(*cache->z);


    /*====================================
      TEST 3: guardarCacheLayer
    ====================================*/

    printf("\n===== TEST 3: Crear cache con A y Z =====\n");

    struct cache *cache2 = inicializarCache();
    guardarCacheLayerA(cache2, *a1);
    guardarCacheLayerZ(cache2, *z1);



    if(cache2 != NULL)
        printf("Cache creada correctamente.\n");
    else
        printf("ERROR al crear cache.\n");

    if(cache2 != NULL)
    {
        printf("\nA de la cache:\n");
        imprimirMatriz(*cache2->a);

        printf("\nZ de la cache:\n");
        imprimirMatriz(*cache2->z);
    }


    /*====================================
      TEST 4: Forward con cache
    ====================================*/

    printf("\n===== TEST 4: Forward con cache =====\n");

    struct matrix *pesos;
    struct matrix *bias;
    struct layer *L;

    crearMatriz(&pesos, 2, 3);
    crearMatriz(&bias, 2, 1);

    float datosPesos[] = {
        1, 0, 1,
        0, 1, 1
    };

    float datosBias[] = {
        1,
        2
    };

    inicializarMatriz(pesos, datosPesos);
    inicializarMatriz(bias, datosBias);

    if(crearLayer(pesos, bias, RELU, &L) == 1)
        printf("Layer creada correctamente.\n");
    else{
        printf("ERROR al crear layer.\n");
        return 1;
    }

    struct matrix *entrada;

    crearMatriz(&entrada, 3, 1);

    float datosEntrada[] = {
        1,
        2,
        3
    };

    inicializarMatriz(entrada, datosEntrada);

    struct matrix *salida;

    crearMatriz(&salida, 2, 1);

    struct cache *cacheForward = forward(entrada, L, salida);

    if(cacheForward != NULL)
        printf("Forward ha generado la cache correctamente.\n");
    else
        printf("ERROR: forward no ha generado la cache.\n");

    if(cacheForward != NULL)
    {
        printf("\nA guardada por forward:\n");
        imprimirMatriz(*cacheForward->a);

        printf("\nZ guardada por forward:\n");
        imprimirMatriz(*cacheForward->z);

        printf("\nSalida:\n");
        imprimirMatriz(*salida);

        printf("\nEsperado:\n");
        printf("A:\n");
        printf("1.000000\n");
        printf("2.000000\n");
        printf("3.000000\n");

        printf("\nZ:\n");
        printf("5.000000\n");
        printf("7.000000\n");

        printf("\nSalida ReLU:\n");
        printf("5.000000\n");
        printf("7.000000\n");
    }


    /*====================================
      TEST 5: Eliminar cache
    ====================================*/

    printf("\n===== TEST 5: Eliminar cache =====\n");

    if(eliminarCacheLayer(&cache) == 1)
        printf("Cache eliminada correctamente.\n");
    else
        printf("ERROR al eliminar cache.\n");

    if(cache == NULL)
        printf("Puntero cache = NULL correctamente.\n");
    else
        printf("ERROR: cache no es NULL.\n");


    /*====================================
      TEST 6: Eliminar segunda cache
    ====================================*/

    printf("\n===== TEST 6: Eliminar segunda cache =====\n");

    if(eliminarCacheLayer(&cache2) == 1)
        printf("Cache eliminada correctamente.\n");
    else
        printf("ERROR al eliminar cache.\n");

    if(cache2 == NULL)
        printf("Puntero cache2 = NULL correctamente.\n");
    else
        printf("ERROR: cache2 no es NULL.\n");


    /*====================================
      Liberar resto de recursos
    ====================================*/

    eliminarMatriz(&a1);
    eliminarMatriz(&z1);

    eliminarMatriz(&entrada);
    eliminarMatriz(&salida);

    eliminarLayer(&L);

    eliminarCacheLayer(&cacheForward);

    printf("\n=====================================\n");
    printf("       TODOS LOS TESTS FINALIZADOS\n");
    printf("=====================================\n");

    return 0;
}