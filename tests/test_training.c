#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "network.h"
#include "layer.h"
#include "matrix.h"
#include "loss.h"
#include "training.h"

#define TOL 1e-5f

static int total = 0;
static int passed = 0;
static int failed = 0;

static void check(const char *name, int ok)
{
    total++;
    if (ok) {
        passed++;
        printf("[PASS] %s\n", name);
    } else {
        failed++;
        printf("[FAIL] %s\n", name);
    }
}

static void print_matrix(const char *name, struct matrix *m)
{
    if (m == NULL) {
        printf("%s = NULL\n", name);
        return;
    }

    printf("%s [%d x %d]\n", name, m->fil, m->col);
    for (int i = 0; i < m->fil; i++) {
        printf("  [ ");
        for (int j = 0; j < m->col; j++) {
            printf("%.9f", m->datos[i * m->col + j]);
            if (j + 1 < m->col)
                printf("  ");
        }
        printf(" ]\n");
    }
}

static float dataset_loss(struct network *red,
                          struct matrix **entradas,
                          struct matrix **esperadas,
                          int n)
{
    float suma = 0.0f;

    for (int i = 0; i < n; i++) {
        struct matrix *salida = forwardRed(red, entradas[i]);

        if (salida == NULL) {
            printf("[ERROR] forwardRed devolvio NULL en la muestra %d.\n", i);
            return NAN;
        }

        suma += mse(*salida, *esperadas[i]);
        eliminarMatriz(&salida);
    }

    return suma / (float)n;
}

static int crear_red_1x1(struct network **red, float peso, float bias)
{
    struct matrix *w = NULL;
    struct matrix *b = NULL;
    struct layer *capa = NULL;

    if (crearRed(red) != 0)
        return 1;

    crearMatriz(&w, 1, 1);
    crearMatriz(&b, 1, 1);

    w->datos[0] = peso;
    b->datos[0] = bias;

    if (crearLayer(w, b, SIGMOID, &capa) != 0) {
        eliminarMatriz(&w);
        eliminarMatriz(&b);
        eliminarRed(red);
        return 1;
    }

    eliminarMatriz(&w);
    eliminarMatriz(&b);

    if (añadirCapa(capa, *red) != 0) {
        eliminarLayer(&capa);
        eliminarRed(red);
        return 1;
    }

    return 0;
}

static void crear_matriz_1x1(struct matrix **m, float valor)
{
    crearMatriz(m, 1, 1);
    (*m)->datos[0] = valor;
}

int main(void)
{
    printf("============================================================\n");
    printf(" TEST DE TRAINING\n");
    printf("============================================================\n");

    /*
     * 1. Una muestra y muchas epocas.
     * Debe producir aprendizaje real: loss_final < loss_inicial.
     */
    printf("\n------------------------------------------------------------\n");
    printf("1. Una muestra, varias epocas\n");
    printf("------------------------------------------------------------\n");

    {
        struct network *red = NULL;
        struct matrix *entradas[1] = { NULL };
        struct matrix *esperadas[1] = { NULL };

        check("crear red 1x1", crear_red_1x1(&red, 0.0f, 0.0f) == 0);

        crear_matriz_1x1(&entradas[0], 1.0f);
        crear_matriz_1x1(&esperadas[0], 1.0f);

        float loss_inicial = dataset_loss(red, entradas, esperadas, 1);

        int rc = training(red, entradas, esperadas, 1.0f, 1, 50);
        check("training devuelve 0", rc == 0);

        float loss_final = dataset_loss(red, entradas, esperadas, 1);

        printf("Loss inicial = %.9f\n", loss_inicial);
        printf("Loss final   = %.9f\n", loss_final);

        check("el loss final es menor que el inicial",
              !isnan(loss_inicial) &&
              !isnan(loss_final) &&
              loss_final < loss_inicial);

        eliminarMatriz(&entradas[0]);
        eliminarMatriz(&esperadas[0]);
        eliminarRed(&red);
    }

    /*
     * 2. Varias muestras.
     * Comprueba que training recibe y procesa un conjunto de entradas,
     * no solamente una.
     */
    printf("\n------------------------------------------------------------\n");
    printf("2. Varias muestras por epoca\n");
    printf("------------------------------------------------------------\n");

    {
        enum { N = 3 };

        struct network *red = NULL;
        struct matrix *entradas[N] = { NULL };
        struct matrix *esperadas[N] = { NULL };

        check("crear red para dataset",
              crear_red_1x1(&red, -1.0f, -1.0f) == 0);

        crear_matriz_1x1(&entradas[0], 0.0f);
        crear_matriz_1x1(&entradas[1], 0.5f);
        crear_matriz_1x1(&entradas[2], 1.0f);

        crear_matriz_1x1(&esperadas[0], 1.0f);
        crear_matriz_1x1(&esperadas[1], 1.0f);
        crear_matriz_1x1(&esperadas[2], 1.0f);

        float loss_inicial = dataset_loss(red, entradas, esperadas, N);
        float peso_inicial = red->capas[0]->pesos->datos[0];
        float bias_inicial = red->capas[0]->bias->datos[0];

        int rc = training(red, entradas, esperadas, 0.5f, N, 100);
        check("training devuelve 0 con varias muestras", rc == 0);

        float loss_final = dataset_loss(red, entradas, esperadas, N);
        float peso_final = red->capas[0]->pesos->datos[0];
        float bias_final = red->capas[0]->bias->datos[0];

        printf("Loss inicial = %.9f\n", loss_inicial);
        printf("Loss final   = %.9f\n", loss_final);
        printf("Peso: %.9f -> %.9f\n", peso_inicial, peso_final);
        printf("Bias: %.9f -> %.9f\n", bias_inicial, bias_final);

        check("el loss final del dataset es menor",
              !isnan(loss_inicial) &&
              !isnan(loss_final) &&
              loss_final < loss_inicial);

        check("al menos un parametro ha cambiado",
              fabsf(peso_final - peso_inicial) > TOL ||
              fabsf(bias_final - bias_inicial) > TOL);

        for (int i = 0; i < N; i++) {
            eliminarMatriz(&entradas[i]);
            eliminarMatriz(&esperadas[i]);
        }

        eliminarRed(&red);
    }

    /*
     * 3. Las epocas deben repetirse.
     * Dos redes parten de los mismos parametros y entrenan el mismo dataset.
     */
    printf("\n------------------------------------------------------------\n");
    printf("3. Diferencia entre pocas y muchas epocas\n");
    printf("------------------------------------------------------------\n");

    {
        enum { N = 3 };

        struct network *red_5 = NULL;
        struct network *red_100 = NULL;

        struct matrix *entradas_5[N] = { NULL };
        struct matrix *esperadas_5[N] = { NULL };
        struct matrix *entradas_100[N] = { NULL };
        struct matrix *esperadas_100[N] = { NULL };

        crear_red_1x1(&red_5, -0.5f, -0.5f);
        crear_red_1x1(&red_100, -0.5f, -0.5f);

        for (int i = 0; i < N; i++) {
            float x = (float)i / 2.0f;

            crear_matriz_1x1(&entradas_5[i], x);
            crear_matriz_1x1(&esperadas_5[i], 1.0f);

            crear_matriz_1x1(&entradas_100[i], x);
            crear_matriz_1x1(&esperadas_100[i], 1.0f);
        }

        int rc5 = training(red_5, entradas_5, esperadas_5,
                           0.5f, N, 5);
        int rc100 = training(red_100, entradas_100, esperadas_100,
                             0.5f, N, 100);

        check("training 5 epocas devuelve 0", rc5 == 0);
        check("training 100 epocas devuelve 0", rc100 == 0);

        float loss_5 = dataset_loss(red_5, entradas_5, esperadas_5, N);
        float loss_100 = dataset_loss(red_100, entradas_100, esperadas_100, N);

        printf("Loss despues de 5 epocas   = %.9f\n", loss_5);
        printf("Loss despues de 100 epocas = %.9f\n", loss_100);

        check("100 epocas mejora este caso respecto a 5",
              !isnan(loss_5) &&
              !isnan(loss_100) &&
              loss_100 < loss_5);

        for (int i = 0; i < N; i++) {
            eliminarMatriz(&entradas_5[i]);
            eliminarMatriz(&esperadas_5[i]);
            eliminarMatriz(&entradas_100[i]);
            eliminarMatriz(&esperadas_100[i]);
        }

        eliminarRed(&red_5);
        eliminarRed(&red_100);
    }

    /*
     * 4. Cero epocas.
     * Con 0 iteraciones no debe haber actualizaciones.
     */
    printf("\n------------------------------------------------------------\n");
    printf("4. Cero epocas: no actualizar\n");
    printf("------------------------------------------------------------\n");

    {
        struct network *red = NULL;
        struct matrix *entrada = NULL;
        struct matrix *esperada = NULL;

        check("crear red para 0 epocas",
              crear_red_1x1(&red, 0.75f, -0.25f) == 0);

        crear_matriz_1x1(&entrada, 1.0f);
        crear_matriz_1x1(&esperada, 1.0f);

        float w_antes = red->capas[0]->pesos->datos[0];
        float b_antes = red->capas[0]->bias->datos[0];

        int rc = training(red, &entrada, &esperada, 0.5f, 1, 0);
        check("training con 0 epocas devuelve 0", rc == 0);

        float w_despues = red->capas[0]->pesos->datos[0];
        float b_despues = red->capas[0]->bias->datos[0];

        check("el peso no cambia con 0 epocas",
              fabsf(w_antes - w_despues) <= TOL);

        check("el bias no cambia con 0 epocas",
              fabsf(b_antes - b_despues) <= TOL);

        eliminarMatriz(&entrada);
        eliminarMatriz(&esperada);
        eliminarRed(&red);
    }

    /*
     * 5. nEntradas=1 pero varias epocas y muestras.
     * Sirve para que una implementación que solo haga una única pasada
     * sea detectada por la mejora posterior.
     */
    printf("\n------------------------------------------------------------\n");
    printf("5. Varias muestras x varias epocas\n");
    printf("------------------------------------------------------------\n");

    {
        enum { N = 4 };

        struct network *red = NULL;
        struct matrix *entradas[N] = { NULL };
        struct matrix *esperadas[N] = { NULL };

        check("crear red final",
              crear_red_1x1(&red, -2.0f, -2.0f) == 0);

        crear_matriz_1x1(&entradas[0], 0.0f);
        crear_matriz_1x1(&entradas[1], 0.25f);
        crear_matriz_1x1(&entradas[2], 0.75f);
        crear_matriz_1x1(&entradas[3], 1.0f);

        for (int i = 0; i < N; i++)
            crear_matriz_1x1(&esperadas[i], 1.0f);

        float inicial = dataset_loss(red, entradas, esperadas, N);

        int rc = training(red, entradas, esperadas,
                          0.25f, N, 200);

        check("training devuelve 0 en entrenamiento completo",
              rc == 0);

        float final = dataset_loss(red, entradas, esperadas, N);

        printf("Loss inicial = %.9f\n", inicial);
        printf("Loss final   = %.9f\n", final);

        check("el entrenamiento completo reduce el loss",
              !isnan(inicial) &&
              !isnan(final) &&
              final < inicial);

        for (int i = 0; i < N; i++) {
            eliminarMatriz(&entradas[i]);
            eliminarMatriz(&esperadas[i]);
        }

        eliminarRed(&red);
    }

    printf("\n============================================================\n");
    printf(" RESUMEN\n");
    printf("============================================================\n");
    printf("Tests totales:  %d\n", total);
    printf("Tests pasados:  %d\n", passed);
    printf("Tests fallados: %d\n", failed);

    if (failed == 0) {
        printf("\nTODOS LOS TESTS DE TRAINING HAN PASADO.\n");
        return 0;
    }

    printf("\nHAY %d TEST(S) FALLADO(S).\n", failed);
    return 1;
}
