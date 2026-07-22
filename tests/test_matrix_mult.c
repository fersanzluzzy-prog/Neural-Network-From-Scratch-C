#include <stdio.h>

#include "matrix.h"

int main(void)
{
    printf("=====================================\n");
    printf(" TESTS MULTIPLICACION MATRICIAL\n");
    printf("=====================================\n");

    struct matrix *A;
    struct matrix *B;
    struct matrix *R;

    /*====================================
      TEST 1: Matriz identidad
    ====================================*/

    printf("\n===== TEST 1: Matriz identidad =====\n");

    crearMatriz(&A,2,2);
    crearMatriz(&B,2,2);
    crearMatriz(&R,2,2);

    float datosA[] = {
        1,2,
        3,4
    };

    float datosB[] = {
        1,0,
        0,1
    };

    inicializarMatriz(A,datosA);
    inicializarMatriz(B,datosB);

    multiplicacionMatricial(A,B,R);

    printf("Resultado:\n");
    imprimirMatriz(*R);

    printf("Esperado:\n");
    printf("1 2\n");
    printf("3 4\n");

    eliminarMatriz(&A);
    eliminarMatriz(&B);
    eliminarMatriz(&R);

    /*====================================
      TEST 2: Matriz x Vector
    ====================================*/

    printf("\n===== TEST 2: Matriz x Vector =====\n");

    crearMatriz(&A,2,3);
    crearMatriz(&B,3,1);
    crearMatriz(&R,2,1);

    float datosA2[] = {
        1,2,3,
        4,5,6
    };

    float datosB2[] = {
        7,
        8,
        9
    };

    inicializarMatriz(A,datosA2);
    inicializarMatriz(B,datosB2);

    multiplicacionMatricial(A,B,R);

    printf("Resultado:\n");
    imprimirMatriz(*R);

    printf("Esperado:\n");
    printf("50\n");
    printf("122\n");

    eliminarMatriz(&A);
    eliminarMatriz(&B);
    eliminarMatriz(&R);

    /*====================================
      TEST 3: Matrices de unos
    ====================================*/

    printf("\n===== TEST 3: Matrices de unos =====\n");

    crearConNumero(&A,3,4,1.0f);
    crearConNumero(&B,4,2,1.0f);
    crearMatriz(&R,3,2);

    multiplicacionMatricial(A,B,R);

    printf("Resultado:\n");
    imprimirMatriz(*R);

    printf("Esperado:\n");
    printf("4 4\n");
    printf("4 4\n");
    printf("4 4\n");

    eliminarMatriz(&A);
    eliminarMatriz(&B);
    eliminarMatriz(&R);

    /*====================================
      TEST 4: Matriz nula
    ====================================*/

    printf("\n===== TEST 4: Matriz nula =====\n");

    crearConNumero(&A,5,3,0.0f);
    crearConNumero(&B,3,7,5.0f);
    crearMatriz(&R,5,7);

    multiplicacionMatricial(A,B,R);

    printf("Resultado:\n");
    imprimirMatriz(*R);

    printf("Esperado: Todo ceros.\n");

    eliminarMatriz(&A);
    eliminarMatriz(&B);
    eliminarMatriz(&R);

    /*====================================
      TEST 5: Caso general
    ====================================*/

    printf("\n===== TEST 5: Caso general =====\n");

    crearMatriz(&A,3,2);
    crearMatriz(&B,2,4);
    crearMatriz(&R,3,4);

    float datosA3[] = {
        1,2,
        3,4,
        5,6
    };

    float datosB3[] = {
        7,8,9,10,
        11,12,13,14
    };

    inicializarMatriz(A,datosA3);
    inicializarMatriz(B,datosB3);

    multiplicacionMatricial(A,B,R);

    printf("Resultado:\n");
    imprimirMatriz(*R);

    printf("Esperado:\n");
    printf("29 32 35 38\n");
    printf("65 72 79 86\n");
    printf("101 112 123 134\n");

    eliminarMatriz(&A);
    eliminarMatriz(&B);
    eliminarMatriz(&R);

    printf("\nTodos los tests finalizados correctamente.\n");

    return 0;
}