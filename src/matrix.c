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
    matriz->datos = malloc(filas*columnas*sizeof(float));

    *resultado = matriz;

}

void eliminarMatriz(struct matrix **matriz){
    free((*matriz)->datos);
    free(*matriz);
    *matriz = NULL;
}

//hecho con IA
/*void imprimirMatriz(
    const char *nombre,
    const struct matrix *m)
{
    int i;
    int j;

    if (m == NULL)
    {
        printf("%s = NULL\n", nombre);
        return;
    }

    printf("%s [%d x %d]\n", nombre, m->fil, m->col);

    if (m->datos == NULL)
    {
        printf("  datos = NULL\n");
        return;
    }

    for (i = 0; i < m->fil; i++)
    {
        printf("  [ ");

        for (j = 0; j < m->col; j++)
        {
            printf("% .8f", m->datos[i * m->col + j]);

            if (j < m->col - 1)
                printf(", ");
        }

        printf(" ]\n");
    }
}*/

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

//devuelve 1 si las dimensiones no son correctas
int transponerMatriz(struct matrix *m1, struct matrix *resultado){
    //comprobacion de que las dimensiones son correctas
    if(resultado->col == m1->fil && resultado->fil == m1->col){
        for (int i=0; i<m1->col; i++){
            for (int j=0; j<m1->fil; j++){
                resultado->datos[i * m1->fil + j] = m1->datos[j * resultado->fil + i];
            }
        }
        return 0;
    }
    else{return 1;}
}

//devuelve 1 si las dimensiones son incorrectas
int multiplicarEscalar(struct matrix *m1, float x, struct matrix *resultado){
    //comprobacion de que las dimensiones son correctas
    if(resultado->col == m1->col && resultado->fil == m1->fil){
        for(int i=0; i < resultado->col * resultado->fil; i++){
            resultado->datos[i] = m1->datos[i] * x;
        }
        return 0;
    }
    else{return 1;}
    
}

//Devuelve 1 si las matrices no tienen las mismas dimensiones
int suma(struct matrix m1, struct matrix m2, struct matrix *resultado){

    if(m1.fil == m2.fil && m1.col == m2.col){
        resultado->col = m1.col;
        resultado->fil = m1.fil;
        for (int i=0; i < resultado->col * resultado->fil; i++){
            resultado->datos[i] = m1.datos[i] + m2.datos[i];
        }
        return 0;
    }
    else {
        printf("Las matrices deben tener las mismas dimensiones.");
        return 1;
    }
}

//m1 - m2. Devuelve 1 si las matrices no tienen las mismas dimensiones
int resta(struct matrix m1, struct matrix m2, struct matrix *resultado){

    if(m1.fil == m2.fil && m1.col == m2.col){
        resultado->col = m1.col;
        resultado->fil = m1.fil;
        for (int i=0; i < resultado->col * resultado->fil; i++){
            resultado->datos[i] = m1.datos[i] - m2.datos[i];
        }
        return 0;
    }
    else {
        printf("Las matrices deben tener las mismas dimensiones.");
        return 1;
    }
}

//m1 x m2. Devuelve 1 si las matrices no tienen las dimensiones adecuadas
int multiplicacionMatricial(struct matrix *m1, struct matrix *m2, struct matrix *resultado){
    if(m1->col == m2->fil){
        resultado->col = m2->col;
        resultado->fil = m1->fil;
        for (int i=0; i < resultado->fil; i++){
            for(int j=0; j< resultado->col; j++){
                float x = 0;
                for(int k=0; k<m1->col; k++){
                    x += *accederPos(m1, i, k) * *accederPos(m2, k, j); 
                } 
                modificarPos(resultado, i, j, x);
            }
        }
        return 0;
    }
    else {
        printf("Las matrices deben tener unas dimensiones adecuadas para multiplicarlas.");
        return 1;
    }
}

//multiplicacion elemento por elemento
int multiplicacionElemPorElem(struct matrix *m1, struct matrix *m2, struct matrix *resultado){
    if(m1->col==m2->col && m1->fil==m2->fil){
        for(int i=0; i<m1->col*m1->fil; i++){
            resultado->datos[i] = m1->datos[i] * m2->datos[i];
        }
        return 0;
    }
    return 1;
}

