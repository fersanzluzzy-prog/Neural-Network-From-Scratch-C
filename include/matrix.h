#ifndef NN_MATRIX
#define NN_MATRIX

struct matrix{
    int fil;
    int col;
    float *datos;
};

void crearMatriz(struct matrix **resultado, int filas, int columnas);

void eliminarMatriz(struct matrix **resultado);

void inicializarMatriz(struct matrix *matriz, float *Ndatos);

float *accederPos(struct matrix *matriz, int fila, int columna);

void modificarPos(struct matrix *matriz, int fila, int columna, float dato);

void crearConNumero(struct matrix **resultado, int filas, int columnas, float numero);

void copiarMatriz(struct matrix *m1, struct matrix *m2); //deep copy, copia m1 en m2

float numeroRandom(float min, float max);

void inicializarRandom(struct matrix *matriz, float min, float max);

int transponerMatriz(struct matrix *m1, struct matrix *resultado);

int multiplicarEscalar(struct matrix *m1, float x, struct matrix *resultado);

int suma(struct matrix m1, struct matrix m2, struct matrix *resultado);

int resta(struct matrix m1, struct matrix m2, struct matrix *resultado);

int multiplicacionMatricial(struct matrix *m1, struct matrix *m2, struct matrix *resultado);

int multiplicacionElemPorElem(struct matrix *m1, struct matrix *m2, struct matrix *resultado);

#endif