#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"

int main() {

    srand(time(NULL));

    printf("===== TEST 1: Crear matriz =====\n");

    struct matrix *A;
    crearMatriz(&A, 3, 3);

    printf("Filas: %d\n", A->fil);
    printf("Columnas: %d\n", A->col);

    printf("\n===== TEST 2: Inicializar matriz =====\n");

    float datos[] = {
        1,2,3,
        4,5,6,
        7,8,9
    };

    inicializarMatriz(A, datos);
    imprimirMatriz(*A);

    printf("\n===== TEST 3: Acceder a un elemento =====\n");

    printf("Elemento (1,2): %.2f\n", *accederPos(A,1,2));

    printf("\n===== TEST 4: Modificar elemento =====\n");

    modificarPos(A,1,2,99.0f);
    imprimirMatriz(*A);

    printf("\n===== TEST 5: Crear con un numero =====\n");

    struct matrix *B;
    crearConNumero(&B,4,2,5.5f);

    imprimirMatriz(*B);

    printf("\n===== TEST 6: Copiar matriz =====\n");

    struct matrix *C;
    crearMatriz(&C,A->fil,A->col);

    copiarMatriz(A,C);

    printf("Matriz original:\n");
    imprimirMatriz(*A);

    printf("Matriz copia:\n");
    imprimirMatriz(*C);

    printf("\n===== TEST 7: Copia profunda =====\n");

    modificarPos(A,0,0,-10.0f);

    printf("Original modificada:\n");
    imprimirMatriz(*A);

    printf("La copia NO debe cambiar:\n");
    imprimirMatriz(*C);

    printf("\n===== TEST 8: Numero aleatorio =====\n");

    for(int i=0;i<10;i++)
        printf("%.4f\n", numeroRandom(-1.0f,1.0f));

    printf("\n===== TEST 9: Inicializacion aleatoria =====\n");

    inicializarRandom(B,-1.0f,1.0f);
    imprimirMatriz(*B);

    printf("\n===== TEST 10: Liberar memoria =====\n");

    eliminarMatriz(&A);
    eliminarMatriz(&B);
    eliminarMatriz(&C);

    printf("A = %p\n",(void*)A);
    printf("B = %p\n",(void*)B);
    printf("C = %p\n",(void*)C);

    return 0;
}