#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"

int main(void)
{
    srand(time(NULL));

    printf("=====================================\n");
    printf("      TESTS MODULO MATRIX\n");
    printf("=====================================\n\n");

    /*====================================
      TEST 1 - Crear matriz
    ====================================*/

    printf("===== TEST 1: Crear matriz =====\n");

    struct matrix *A;
    crearMatriz(&A, 3, 3);

    printf("Filas: %d\n", A->fil);
    printf("Columnas: %d\n\n", A->col);

    /*====================================
      TEST 2 - Inicializar matriz
    ====================================*/

    printf("===== TEST 2: Inicializar matriz =====\n");

    float datos[] = {
        1,2,3,
        4,5,6,
        7,8,9
    };

    inicializarMatriz(A, datos);
    imprimirMatriz(*A);

    /*====================================
      TEST 3 - Acceder a un elemento
    ====================================*/

    printf("\n===== TEST 3: Acceder elemento =====\n");

    printf("(0,0) = %.2f\n", *accederPos(A,0,0));
    printf("(1,2) = %.2f\n", *accederPos(A,1,2));
    printf("(2,1) = %.2f\n", *accederPos(A,2,1));

    /*====================================
      TEST 4 - Modificar elemento
    ====================================*/

    printf("\n===== TEST 4: Modificar elemento =====\n");

    modificarPos(A,1,2,99.0f);

    imprimirMatriz(*A);

    /*====================================
      TEST 5 - Crear con numero
    ====================================*/

    printf("\n===== TEST 5: Crear con numero =====\n");

    struct matrix *B;

    crearConNumero(&B,4,2,5.5f);

    imprimirMatriz(*B);

    /*====================================
      TEST 6 - Copiar matriz
    ====================================*/

    printf("\n===== TEST 6: Copiar matriz =====\n");

    struct matrix *C;

    crearMatriz(&C,A->fil,A->col);

    copiarMatriz(A,C);

    printf("Original:\n");
    imprimirMatriz(*A);

    printf("Copia:\n");
    imprimirMatriz(*C);

    /*====================================
      TEST 7 - Copia profunda
    ====================================*/

    printf("\n===== TEST 7: Copia profunda =====\n");

    modificarPos(A,0,0,-10);

    printf("Original modificada:\n");
    imprimirMatriz(*A);

    printf("La copia debe permanecer igual:\n");
    imprimirMatriz(*C);

    /*====================================
      TEST 8 - Numero aleatorio
    ====================================*/

    printf("\n===== TEST 8: Numero aleatorio =====\n");

    for(int i=0;i<10;i++)
        printf("%.4f\n",numeroRandom(-1.0f,1.0f));

    /*====================================
      TEST 9 - Inicializacion aleatoria
    ====================================*/

    printf("\n===== TEST 9: Inicializacion aleatoria =====\n");

    inicializarRandom(B,-1.0f,1.0f);

    imprimirMatriz(*B);

    /*====================================
      TEST 10 - Transpuesta
    ====================================*/

    printf("\n===== TEST 10: Transpuesta =====\n");

    struct matrix *T;

    crearMatriz(&T,A->col,A->fil);

    printf("A: %d x %d\n", A->fil, A->col);
    printf("T: %d x %d\n", T->fil, T->col);
    transponerMatriz(A,T);

    printf("Original:\n");
    imprimirMatriz(*A);

    printf("Transpuesta:\n");
    imprimirMatriz(*T);

    /*====================================
      TEST 11 - Multiplicar escalar
    ====================================*/

    printf("\n===== TEST 11: Multiplicar escalar =====\n");

    struct matrix *Escalar;

    crearMatriz(&Escalar,A->fil,A->col);

    multiplicarEscalar(A,2.0f,Escalar);

    imprimirMatriz(*Escalar);

    /*====================================
      TEST 12 - Suma
    ====================================*/

    printf("\n===== TEST 12: Suma =====\n");

    struct matrix *Ones;
    struct matrix *Resultado;

    crearConNumero(&Ones,A->fil,A->col,1.0f);
    crearMatriz(&Resultado,A->fil,A->col);

    if(suma(*A,*Ones,Resultado)==1)
        imprimirMatriz(*Resultado);
    else
        printf("ERROR EN SUMA\n");

    /*====================================
      TEST 13 - Resta
    ====================================*/

    printf("\n===== TEST 13: Resta =====\n");

    if(resta(*Resultado,*Ones,Resultado)==1)
        imprimirMatriz(*Resultado);
    else
        printf("ERROR EN RESTA\n");

    /*====================================
      TEST 14 - Multiplicacion matricial
    ====================================*/

    printf("\n===== TEST 14: Multiplicacion matricial =====\n");

    float datosA[]={
        1,2,
        3,4
    };

    float datosB[]={
        5,6,
        7,8
    };

    struct matrix *M1;
    struct matrix *M2;
    struct matrix *MR;

    crearMatriz(&M1,2,2);
    crearMatriz(&M2,2,2);
    crearMatriz(&MR,2,2);

    inicializarMatriz(M1,datosA);
    inicializarMatriz(M2,datosB);

    multiplicacionMatricial(M1,M2,MR);

    printf("Matriz A:\n");
    imprimirMatriz(*M1);

    printf("Matriz B:\n");
    imprimirMatriz(*M2);

    printf("Resultado esperado:\n");
    printf("19 22\n");
    printf("43 50\n\n");

    printf("Resultado obtenido:\n");
    imprimirMatriz(*MR);

    /*====================================
      TEST 15 - Liberar memoria
    ====================================*/

    printf("\n===== TEST 15: Liberar memoria =====\n");

    eliminarMatriz(&A);
    eliminarMatriz(&B);
    eliminarMatriz(&C);
    eliminarMatriz(&T);
    eliminarMatriz(&Escalar);
    eliminarMatriz(&Ones);
    eliminarMatriz(&Resultado);
    eliminarMatriz(&M1);
    eliminarMatriz(&M2);
    eliminarMatriz(&MR);

    printf("A  = %p\n",(void*)A);
    printf("B  = %p\n",(void*)B);
    printf("C  = %p\n",(void*)C);
    printf("T  = %p\n",(void*)T);
    printf("Esc= %p\n",(void*)Escalar);
    printf("On = %p\n",(void*)Ones);
    printf("R  = %p\n",(void*)Resultado);
    printf("M1 = %p\n",(void*)M1);
    printf("M2 = %p\n",(void*)M2);
    printf("MR = %p\n",(void*)MR);

    printf("\nTodos los tests finalizados.\n");

    return 0;
}