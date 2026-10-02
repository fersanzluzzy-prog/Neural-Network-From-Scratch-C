/*
 * test_descenso_gradiente.c
 *
 * Tests de int descensoDeGradiente(struct network *red, float factorAprendizaje)
 *
 * Suposiciones del contrato usadas por este test:
 *   - devuelve 0 si todo va bien y 1 si hay error
 *   - actualiza pesos y bias IN-PLACE
 *   - regla de actualizacion:
 *         parametro_nuevo = parametro_antiguo - lr * gradiente
 *   - no hay batches
 *   - pesos tienen dimensiones nSalidas x nEntradas
 *   - bias tiene dimensiones nSalidas x 1
 *   - backprop deja dL_dw y dL_db en cada capa
 *
 * Este test NO usa backpropRed para generar gradientes. Los fija manualmente
 * para poder comprobar descensoDeGradiente de forma aislada y exacta.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "network.h"
#include "layer.h"
#include "matrix.h"

#define ABS_TOL 1e-6f
#define REL_TOL 1e-6f

static int total_tests = 0;
static int passed_tests = 0;
static int failed_tests = 0;

static void line(void)
{
    printf("------------------------------------------------------------\n");
}

static void section(const char *name)
{
    printf("\n");
    line();
    printf("%s\n", name);
    line();
}

static void pass(const char *name)
{
    total_tests++;
    passed_tests++;
    printf("[PASS] %s\n", name);
}

static void fail(const char *name, const char *reason)
{
    total_tests++;
    failed_tests++;
    printf("[FAIL] %s\n", name);
    if (reason != NULL)
        printf("       %s\n", reason);
}

static int almost_equal(float a, float b)
{
    float diff = fabsf(a - b);
    float scale = fmaxf(fabsf(a), fabsf(b));
    return diff <= ABS_TOL + REL_TOL * scale;
}

static struct matrix *make_matrix(int fil, int col, const float *datos)
{
    struct matrix *m = NULL;
    crearMatriz(&m, fil, col);
    if (m == NULL)
        return NULL;

    inicializarMatriz(m, (float *)datos);
    return m;
}

static struct layer *make_layer(int wFil, int wCol, const float *w,
                                int bFil, int bCol, const float *b)
{
    struct matrix *mw = make_matrix(wFil, wCol, w);
    struct matrix *mb = make_matrix(bFil, bCol, b);
    struct layer *layer = NULL;

    if (mw == NULL || mb == NULL) {
        if (mw != NULL) eliminarMatriz(&mw);
        if (mb != NULL) eliminarMatriz(&mb);
        return NULL;
    }

    int rc = crearLayer(mw, mb, RELU, &layer);

    eliminarMatriz(&mw);
    eliminarMatriz(&mb);

    if (rc != 0 || layer == NULL) {
        if (layer != NULL) eliminarLayer(&layer);
        return NULL;
    }

    return layer;
}

static struct network *make_network(void)
{
    struct network *red = NULL;

    if (crearRed(&red) != 0 || red == NULL)
        return NULL;

    return red;
}

static int add_layer(struct network *red, struct layer *layer)
{
    return añadirCapa(layer, red) == 0;
}

static void print_matrix(const char *name, const struct matrix *m)
{
    if (m == NULL) {
        printf("       %s = NULL\n", name);
        return;
    }

    printf("       %s [%d x %d]\n", name, m->fil, m->col);
    for (int i = 0; i < m->fil; ++i) {
        printf("       [");
        for (int j = 0; j < m->col; ++j) {
            float *p = accederPos((struct matrix *)m, i, j);
            if (p != NULL)
                printf(" % .9f", *p);
            else
                printf(" <NULL>");
        }
        printf(" ]\n");
    }
}

static int check_matrix_exact(const char *name,
                              const struct matrix *actual,
                              int expectedFil,
                              int expectedCol,
                              const float *expected)
{
    if (actual == NULL) {
        char reason[128];
        snprintf(reason, sizeof(reason),
                 "Matriz NULL. Esperada [%d x %d].",
                 expectedFil, expectedCol);
        fail(name, reason);
        return 0;
    }

    if (actual->fil != expectedFil || actual->col != expectedCol) {
        char reason[256];
        snprintf(reason, sizeof(reason),
                 "Dimensiones incorrectas. Esperado [%d x %d], obtenido [%d x %d].",
                 expectedFil, expectedCol, actual->fil, actual->col);
        fail(name, reason);
        print_matrix("obtenida", actual);
        return 0;
    }

    int ok = 1;

    for (int i = 0; i < expectedFil; ++i) {
        for (int j = 0; j < expectedCol; ++j) {
            float *p = accederPos((struct matrix *)actual, i, j);
            float e = expected[i * expectedCol + j];

            if (p == NULL || !isfinite(*p) || !almost_equal(*p, e)) {
                ok = 0;
                printf("       Error en [%d,%d]: esperado = % .9f, obtenido = ",
                       i, j, e);
                if (p == NULL)
                    printf("<NULL>\n");
                else
                    printf("% .9f, |error| = %.9g\n", *p, fabsf(*p - e));
            }
        }
    }

    if (!ok) {
        fail(name, "La matriz no coincide con los valores esperados.");
        print_matrix("obtenida", actual);
        return 0;
    }

    pass(name);
    return 1;
}

static int check_code(const char *name, int actual, int expected)
{
    if (actual == expected) {
        pass(name);
        return 1;
    }

    char reason[128];
    snprintf(reason, sizeof(reason), "Esperado %d, obtenido %d.", expected, actual);
    fail(name, reason);
    return 0;
}

static int allocate_gradients(struct layer *layer,
                              int wFil, int wCol,
                              const float *dw,
                              int bFil, int bCol,
                              const float *db)
{
    if (layer == NULL)
        return 0;

    layer->gradientes = malloc(sizeof(struct gradientesLayer));
    if (layer->gradientes == NULL)
        return 0;

    layer->gradientes->dL_dw = make_matrix(wFil, wCol, dw);
    layer->gradientes->dL_db = make_matrix(bFil, bCol, db);
    layer->gradientes->dL_dx = NULL;

    if (layer->gradientes->dL_dw == NULL || layer->gradientes->dL_db == NULL)
        return 0;

    return 1;
}

static void test_single_layer_exact(void)
{
    section("1. Descenso de gradiente: una capa, valores exactos");

    const float W[] = {
         1.0f,  2.0f,
         3.0f,  4.0f
    };
    const float B[] = {
         0.5f,
        -1.0f
    };
    const float dW[] = {
         0.1f, -0.2f,
         0.3f,  0.4f
    };
    const float dB[] = {
         0.5f,
        -0.6f
    };

    const float expectedW[] = {
         0.99f, 2.02f,
         2.97f, 3.96f
    };
    const float expectedB[] = {
         0.45f,
        -0.94f
    };

    struct network *red = make_network();
    struct layer *layer = make_layer(2, 2, W, 2, 1, B);

    if (red == NULL || layer == NULL) {
        fail("preparacion del test", "No se pudo crear la red/capa.");
        if (layer != NULL) eliminarLayer(&layer);
        if (red != NULL) eliminarRed(&red);
        return;
    }

    if (!add_layer(red, layer)) {
        fail("añadirCapa", "No se pudo añadir la capa al test.");
        eliminarLayer(&layer);
        eliminarRed(&red);
        return;
    }

    if (!allocate_gradients(red->capas[0], 2, 2, dW, 2, 1, dB)) {
        fail("preparacion de gradientes", "No se pudieron crear los gradientes.");
        eliminarRed(&red);
        return;
    }

    check_code("descensoDeGradiente devuelve 0",
               descensoDeGradiente(red, 0.1f), 0);

    check_matrix_exact("W despues del descenso",
                       red->capas[0]->pesos, 2, 2, expectedW);

    check_matrix_exact("bias despues del descenso",
                       red->capas[0]->bias, 2, 1, expectedB);
    eliminarRed(&red);
}

static void test_two_layers_exact(void)
{
    section("2. Descenso de gradiente: dos capas");

    const float W1[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    const float B1[] = {
        1.0f,
        2.0f
    };
    const float dW1[] = {
         1.0f, -2.0f,
        -3.0f,  4.0f
    };
    const float dB1[] = {
         0.5f,
        -1.0f
    };

    const float W2[] = {
         5.0f, 6.0f
    };
    const float B2[] = {
        -2.0f
    };
    const float dW2[] = {
        -0.5f, 2.0f
    };
    const float dB2[] = {
         3.0f
    };

    const float expectedW1[] = {
         0.9f,  2.2f,
         3.3f,  3.6f
    };
    const float expectedB1[] = {
         0.95f,
         2.1f
    };
    const float expectedW2[] = {
         5.05f, 5.8f
    };
    const float expectedB2[] = {
        -2.3f
    };

    struct network *red = make_network();
    struct layer *l1 = make_layer(2, 2, W1, 2, 1, B1);
    struct layer *l2 = make_layer(1, 2, W2, 1, 1, B2);

    if (red == NULL || l1 == NULL || l2 == NULL) {
        fail("preparacion del test", "No se pudo crear la red/capas.");
        if (l1 != NULL) eliminarLayer(&l1);
        if (l2 != NULL) eliminarLayer(&l2);
        if (red != NULL) eliminarRed(&red);
        return;
    }

    if (!add_layer(red, l1)) {
        fail("añadirCapa 1", "No se pudo añadir la primera capa.");
        eliminarLayer(&l1);
        eliminarLayer(&l2);
        eliminarRed(&red);
        return;
    }

    if (!add_layer(red, l2)) {
        fail("añadirCapa 2", "No se pudo añadir la segunda capa.");
        eliminarLayer(&l2);
        eliminarRed(&red);
        return;
    }

    if (!allocate_gradients(red->capas[0], 2, 2, dW1, 2, 1, dB1) ||
        !allocate_gradients(red->capas[1], 1, 2, dW2, 1, 1, dB2)) {
        fail("preparacion de gradientes", "No se pudieron crear los gradientes.");
        eliminarRed(&red);
        return;
    }

    check_code("descensoDeGradiente devuelve 0",
               descensoDeGradiente(red, 0.1f), 0);

    check_matrix_exact("capa 1 W",
                       red->capas[0]->pesos, 2, 2, expectedW1);
    check_matrix_exact("capa 1 bias",
                       red->capas[0]->bias, 2, 1, expectedB1);
    check_matrix_exact("capa 2 W",
                       red->capas[1]->pesos, 1, 2, expectedW2);
    check_matrix_exact("capa 2 bias",
                       red->capas[1]->bias, 1, 1, expectedB2);

    eliminarRed(&red);
}

static void test_learning_rate_zero(void)
{
    section("3. Factor de aprendizaje 0: no debe cambiar parametros");

    const float W[] = {
        -2.0f, 4.0f,
         1.5f, 3.0f
    };
    const float B[] = {
         7.0f,
        -5.0f
    };
    const float dW[] = {
        100.0f, -200.0f,
        300.0f, 400.0f
    };
    const float dB[] = {
        500.0f,
       -600.0f
    };

    struct network *red = make_network();
    struct layer *layer = make_layer(2, 2, W, 2, 1, B);

    if (red == NULL || layer == NULL) {
        fail("preparacion del test", "No se pudo crear la red/capa.");
        if (layer != NULL) eliminarLayer(&layer);
        if (red != NULL) eliminarRed(&red);
        return;
    }

    if (!add_layer(red, layer)) {
        fail("añadirCapa", "No se pudo añadir la capa al test.");
        eliminarLayer(&layer);
        eliminarRed(&red);
        return;
    }

    if (!allocate_gradients(red->capas[0], 2, 2, dW, 2, 1, dB)) {
        fail("preparacion de gradientes", "No se pudieron crear los gradientes.");
        eliminarRed(&red);
        return;
    }

    check_code("factor 0 devuelve 0",
               descensoDeGradiente(red, 0.0f), 0);

    check_matrix_exact("W no cambia con factor 0", red->capas[0]->pesos,
                       2, 2, W);
    check_matrix_exact("bias no cambia con factor 0", red->capas[0]->bias,
                       2, 1, B);

    eliminarRed(&red);
}

static void test_positive_and_negative_gradients(void)
{
    section("4. Signos: positivos y negativos");

    const float W[] = { 10.0f, 20.0f, 30.0f };
    const float B[] = { 40.0f, 50.0f, 60.0f };
    const float dW[] = { 2.0f, -3.0f, 4.0f };
    const float dB[] = { -5.0f, 6.0f, -7.0f };

    const float expectedW[] = { 9.0f, 21.5f, 28.0f };
    const float expectedB[] = { 42.5f, 47.0f, 63.5f };

    struct network *red = make_network();
    struct layer *layer = make_layer(3, 1, W, 3, 1, B);

    if (red == NULL || layer == NULL) {
        fail("preparacion del test", "No se pudo crear la red/capa.");
        if (layer != NULL) eliminarLayer(&layer);
        if (red != NULL) eliminarRed(&red);
        return;
    }

    if (!add_layer(red, layer)) {
        fail("añadirCapa", "No se pudo añadir la capa al test.");
        eliminarLayer(&layer);
        eliminarRed(&red);
        return;
    }

    if (!allocate_gradients(red->capas[0], 3, 1, dW, 3, 1, dB)) {
        fail("preparacion de gradientes", "No se pudieron crear los gradientes.");
        eliminarRed(&red);
        return;
    }

    check_code("descenso con gradientes de ambos signos devuelve 0",
               descensoDeGradiente(red, 0.5f), 0);

    check_matrix_exact("W con signos mixtos", red->capas[0]->pesos,
                       3, 1, expectedW);
    check_matrix_exact("bias con signos mixtos", red->capas[0]->bias,
                       3, 1, expectedB);

    eliminarRed(&red);
}

static void test_repeated_updates(void)
{
    section("5. Dos pasos consecutivos: actualizacion acumulativa");

    const float W[] = { 1.0f, 2.0f };
    const float B[] = { 3.0f };
    const float dW[] = { 0.5f, -1.0f };
    const float dB[] = { 2.0f };

    struct network *red = make_network();
    struct layer *layer = make_layer(1, 2, W, 1, 1, B);

    if (red == NULL || layer == NULL) {
        fail("preparacion del test", "No se pudo crear la red/capa.");
        if (layer != NULL) eliminarLayer(&layer);
        if (red != NULL) eliminarRed(&red);
        return;
    }

    if (!add_layer(red, layer)) {
        fail("añadirCapa", "No se pudo añadir la capa al test.");
        eliminarLayer(&layer);
        eliminarRed(&red);
        return;
    }

    if (!allocate_gradients(red->capas[0], 1, 2, dW, 1, 1, dB)) {
        fail("preparacion de gradientes", "No se pudieron crear los gradientes.");
        eliminarRed(&red);
        return;
    }

    check_code("primer paso devuelve 0",
               descensoDeGradiente(red, 0.2f), 0);

    {
        const float expectedW1[] = { 0.9f, 2.2f };
        const float expectedB1[] = { 2.6f };
        check_matrix_exact("W tras primer paso", red->capas[0]->pesos,
                           1, 2, expectedW1);
        check_matrix_exact("bias tras primer paso", red->capas[0]->bias,
                           1, 1, expectedB1);
    }

    check_code("segundo paso devuelve 0",
               descensoDeGradiente(red, 0.2f), 0);

    {
        const float expectedW2[] = { 0.8f, 2.4f };
        const float expectedB2[] = { 2.2f };
        check_matrix_exact("W tras segundo paso", red->capas[0]->pesos,
                           1, 2, expectedW2);
        check_matrix_exact("bias tras segundo paso", red->capas[0]->bias,
                           1, 1, expectedB2);
    }

    eliminarRed(&red);
}

int main(void)
{
    printf("============================================================\n");
    printf(" TEST DE DESCENSO DE GRADIENTE\n");
    printf(" Regla esperada: parametro -= factorAprendizaje * gradiente\n");
    printf("============================================================\n");

    test_single_layer_exact();
    test_two_layers_exact();
    test_learning_rate_zero();
    test_positive_and_negative_gradients();
    test_repeated_updates();

    printf("\n");
    line();
    printf("RESUMEN\n");
    line();
    printf("Tests totales:  %d\n", total_tests);
    printf("Tests pasados:  %d\n", passed_tests);
    printf("Tests fallados: %d\n", failed_tests);

    if (failed_tests == 0) {
        printf("\nTODOS LOS TESTS DE DESCENSO DE GRADIENTE HAN PASADO.\n");
        return 0;
    }

    printf("\nHAY %d TEST(S) FALLADO(S).\n", failed_tests);
    return 1;
}
