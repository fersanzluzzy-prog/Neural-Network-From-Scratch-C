#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "layer.h"
#include "network.h"

int main(void)
{
    printf("=====================================\n");
    printf("      TESTS BACKPROPAGATION RED\n");
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


    /*====================================
      TEST 2: Crear Layer 1
    ====================================*/

    printf("\n===== TEST 2: Crear Layer 1 =====\n");

    struct matrix *pesos1;
    struct matrix *bias1;
    struct layer *layer1;

    crearMatriz(&pesos1, 2, 2);
    crearMatriz(&bias1, 2, 1);

    float datosPesos1[] = {
        1, 2,
        3, 4
    };

    float datosBias1[] = {
        1,
        1
    };

    inicializarMatriz(pesos1, datosPesos1);
    inicializarMatriz(bias1, datosBias1);

    if(crearLayer(pesos1, bias1, RELU, &layer1) == 1)
        printf("Layer 1 creada correctamente.\n");
    else{
        printf("ERROR al crear Layer 1.\n");
        eliminarRed(&red);
        return 1;
    }

    if(añadirCapa(layer1, red) == 1)
        printf("Layer 1 añadida correctamente.\n");
    else{
        printf("ERROR al añadir Layer 1.\n");
        eliminarRed(&red);
        return 1;
    }


    /*====================================
      TEST 3: Crear Layer 2
    ====================================*/

    printf("\n===== TEST 3: Crear Layer 2 =====\n");

    struct matrix *pesos2;
    struct matrix *bias2;
    struct layer *layer2;

    crearMatriz(&pesos2, 1, 2);
    crearMatriz(&bias2, 1, 1);

    float datosPesos2[] = {
        2, 1
    };

    float datosBias2[] = {
        1
    };

    inicializarMatriz(pesos2, datosPesos2);
    inicializarMatriz(bias2, datosBias2);

    if(crearLayer(pesos2, bias2, RELU, &layer2) == 1)
        printf("Layer 2 creada correctamente.\n");
    else{
        printf("ERROR al crear Layer 2.\n");
        eliminarRed(&red);
        return 1;
    }

    if(añadirCapa(layer2, red) == 1)
        printf("Layer 2 añadida correctamente.\n");
    else{
        printf("ERROR al añadir Layer 2.\n");
        eliminarRed(&red);
        return 1;
    }


    /*====================================
      TEST 4: Entrada
    ====================================*/

    printf("\n===== TEST 4: Entrada =====\n");

    struct matrix *entrada;

    crearMatriz(&entrada, 2, 1);

    float datosEntrada[] = {
        1,
        2
    };

    inicializarMatriz(entrada, datosEntrada);

    printf("Entrada:\n");
    imprimirMatriz(*entrada);


    /*====================================
      TEST 5: Forward
    ====================================*/

    printf("\n===== TEST 5: Forward =====\n");

    struct matrix *salida;

    salida = forwardRed(red, entrada);

    if(salida != NULL)
        printf("Forward realizado correctamente.\n");
    else{
        printf("ERROR en forward.\n");
        eliminarMatriz(&entrada);
        eliminarRed(&red);
        return 1;
    }

    printf("\nSalida de la red:\n");
    imprimirMatriz(*salida);


    /*====================================
      TEST 6: Resultado correcto
    ====================================*/

    printf("\n===== TEST 6: Resultado correcto =====\n");

    struct matrix *esperada;

    crearMatriz(&esperada, 1, 1);

    float datosEsperada[] = {
        20
    };

    inicializarMatriz(esperada, datosEsperada);

    printf("Esperada:\n");
    imprimirMatriz(*esperada);


    /*====================================
      TEST 7: Backpropagation
    ====================================*/

    printf("\n===== TEST 7: Backpropagation =====\n");

    if(backpropRed(red, *salida, *esperada) == 1)
        printf("Backpropagation realizado correctamente.\n");
    else{
        printf("ERROR en backpropagation.\n");
        eliminarMatriz(&entrada);
        eliminarMatriz(&salida);
        eliminarMatriz(&esperada);
        eliminarRed(&red);
        return 1;
    }


    /*====================================
      TEST 8: Gradientes Layer 1
    ====================================*/

    printf("\n===== TEST 8: Gradientes Layer 1 =====\n");

    printf("\ndL/dW Layer 1:\n");
    imprimirMatriz(*red->capas[0]->gradientes->dL_dw);

    printf("\ndL/db Layer 1:\n");
    imprimirMatriz(*red->capas[0]->gradientes->dL_db);

    printf("\ndL/dz Layer 1:\n");
    imprimirMatriz(*red->capas[0]->gradientes->dL_dz);


    /*====================================
      TEST 9: Gradientes Layer 2
    ====================================*/

    printf("\n===== TEST 9: Gradientes Layer 2 =====\n");

    printf("\ndL/dW Layer 2:\n");
    imprimirMatriz(*red->capas[1]->gradientes->dL_dw);

    printf("\ndL/db Layer 2:\n");
    imprimirMatriz(*red->capas[1]->gradientes->dL_db);

    printf("\ndL/dz Layer 2:\n");
    imprimirMatriz(*red->capas[1]->gradientes->dL_dz);


    /*====================================
      TEST 10: Comprobar que existen
    ====================================*/

    printf("\n===== TEST 10: Comprobar gradientes =====\n");

    if(red->capas[0]->gradientes != NULL &&
       red->capas[0]->gradientes->dL_dw != NULL &&
       red->capas[0]->gradientes->dL_db != NULL &&
       red->capas[0]->gradientes->dL_dz != NULL)
    {
        printf("Gradientes Layer 1 creados correctamente.\n");
    }
    else
    {
        printf("ERROR en los gradientes Layer 1.\n");
    }

    if(red->capas[1]->gradientes != NULL &&
       red->capas[1]->gradientes->dL_dw != NULL &&
       red->capas[1]->gradientes->dL_db != NULL &&
       red->capas[1]->gradientes->dL_dz != NULL)
    {
        printf("Gradientes Layer 2 creados correctamente.\n");
    }
    else
    {
        printf("ERROR en los gradientes Layer 2.\n");
    }


    /*====================================
      Liberar memoria
    ====================================*/

    eliminarMatriz(&entrada);
    eliminarMatriz(&salida);
    eliminarMatriz(&esperada);

    eliminarRed(&red);

    printf("\n=====================================\n");
    printf("       TESTS FINALIZADOS\n");
    printf("=====================================\n");

    return 0;
}