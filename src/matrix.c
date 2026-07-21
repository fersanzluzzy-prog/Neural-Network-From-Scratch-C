#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

void crearMatriz(struct matrix **resultado, int filas, int columnas){

    struct matrix *matriz = malloc(sizeof(struct matrix));
    if (matriz == NULL) {
        printf("Error al reservar memoria\n");
        return;
    }
    matriz->fil = filas;
    matriz->col = columnas;
    matriz->datos = NULL;

    *resultado = matriz;

}

void eliminarMatriz(struct matrix **matriz){
    free((*matriz)->datos);
    free(*matriz);
    *matriz = NULL;
}

void imprimirMatriz(struct matrix matriz){
    int filas = matriz.fil;
    int columnas = matriz.col;
    
    for(int i=0; i < filas; i++){
        
        for(int j=0; j < columnas; j++){
            printf("%f, ", matriz.datos[columnas*(i) + j ]);
        }
        printf("\n");
    }
}

void inicializarMatriz(struct matrix *matriz, float *Ndatos){
    int filas = matriz->fil;
    int columnas = matriz->col;
    matriz->datos = malloc(filas*columnas*sizeof(float));
    
    for(int i=0; i<filas*columnas; i++){
        matriz->datos[i] = Ndatos[i];
    }

}

//NULL si se dan filas o columnas invalidas
float *accederPos(struct matrix *matriz, int fila, int columna){
    if(fila<0 || fila>=matriz->fil || columna<0 || columna>=matriz->col){return NULL;}
    int nColumnas = matriz->col;
    int indice = nColumnas*fila+columna;
    return &matriz->datos[indice];
}

void modificarPos(struct matrix *matriz, int fila, int columna, float dato){
    if(fila<0 || fila>=matriz->fil || columna<0 || columna>=matriz->col){return;}
    float *datoX = accederPos(matriz, fila, columna);
    *datoX = dato;
}

void crearConNumero(struct matrix **resultado, int filas, int columnas, float numero){
    crearMatriz(resultado, filas, columnas);
    float *datos = malloc(filas*columnas*sizeof(float));
    for(int i=0; i<filas*columnas; i++){
        datos[i] = numero;
    }
    inicializarMatriz(*resultado, datos);
}

//copia m1 en m2
void copiarMatriz(struct matrix *m1, struct matrix *m2){

    m2->fil = m1->fil;
    m2->col = m1->col;
    m2->datos = malloc(m2->fil * m2->col * sizeof(float));

    for(int i=0; i<m2->fil * m2->col; i++){
        m2->datos[i] = m1->datos[i];
    }
}

float numeroRandom(float min, float max){
    return (((float) rand() / RAND_MAX) * (max - min)) + min ;
}

void inicializarRandom(struct matrix *matriz, float min, float max){    
    for(int i=0; i < matriz->col * matriz->fil; i++){
        matriz->datos[i] = numeroRandom(min, max);
    }
}






