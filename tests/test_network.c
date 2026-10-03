/*
 * test_network.c
 *
 * Test exhaustivo de la API publica de network.h, usando tambien el estado
 * publico de network/layer/cache/gradientes para poder inspeccionar resultados.
 *
 * Suposiciones usadas (segun el contrato indicado):
 *   - una matriz de entrada no lleva batches: N x 1
 *   - pesos: nSalidas x nEntradas
 *   - bias: nSalidas x 1
 *   - crearLayer hace copia profunda de pesos y bias
 *   - añadirCapa transfiere la propiedad de la capa a la red
 *   - añadirCapa devuelve 0 si puede anadir la capa y 1 si falla
 *   - eliminarRed libera capas, caches y gradientes de la red
 *   - forwardRed devuelve una matriz nueva que el llamador debe liberar
 *   - cache[i]->z es la preactivacion y cache[i]->a es la activacion
 *   - backpropRed deja dL_dw y dL_db en todas las capas; dL_dx solo en
 *     capas que NO sean la primera
 *   - se usa MSE con 1/(2n) sum((real-esperada)^2). Los tests exactos usan
 *     una unica salida para evitar ambiguedad en el factor 1/n.
 *
 * El gradient checking usa la propia mse() como funcion de perdida y por
 * tanto valida que backpropRed sea consistente con la implementacion real de
 * la perdida, independientemente de la convencion de signo/factor interno.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "network.h"
#include "layer.h"
#include "matrix.h"
#include "activation.h"
#include "loss.h"

#define ABS_TOL 1e-5f
#define REL_TOL 1e-4f
#define GRAD_ABS_TOL 2e-3f
#define GRAD_REL_TOL 2e-3f
#define FD_EPS 1e-3f

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

static void result_ok(const char *name)
{
    total_tests++;
    passed_tests++;
    printf("[PASS] %s\n", name);
}

static void result_fail(const char *name, const char *reason)
{
    total_tests++;
    failed_tests++;
    printf("[FAIL] %s\n", name);
    if (reason && reason[0] != '\0') {
        printf("       %s\n", reason);
    }
}

static int almost_equal(float a, float b, float abs_tol, float rel_tol)
{
    float diff = fabsf(a - b);
    float scale = fmaxf(fabsf(a), fabsf(b));
    return diff <= abs_tol + rel_tol * scale;
}

static void print_matrix(const char *label, const struct matrix *m)
{
    if (!m) {
        printf("       %s = <NULL>\n", label);
        return;
    }

    printf("       %s [%d x %d]\n", label, m->fil, m->col);
    for (int i = 0; i < m->fil; ++i) {
        printf("       [");
        for (int j = 0; j < m->col; ++j) {
            float *p = accederPos((struct matrix *)m, i, j);
            if (p) {
                printf(" % .9f", *p);
            } else {
                printf(" <NULL>");
            }
        }
        printf(" ]\n");
    }
}

static int check_matrix(const char *name,
                        const struct matrix *actual,
                        int rows,
                        int cols,
                        const float *expected,
                        float abs_tol,
                        float rel_tol)
{
    char reason[256];

    if (!actual) {
        snprintf(reason, sizeof(reason), "Matriz obtenida = NULL; se esperaba [%d x %d].", rows, cols);
        result_fail(name, reason);
        return 0;
    }

    if (actual->fil != rows || actual->col != cols) {
        snprintf(reason, sizeof(reason),
                 "Dimensiones incorrectas. Esperado [%d x %d], obtenido [%d x %d].",
                 rows, cols, actual->fil, actual->col);
        result_fail(name, reason);
        print_matrix("obtenida", actual);
        return 0;
    }

    float max_abs = 0.0f;
    int bad = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float *p = accederPos((struct matrix *)actual, i, j);
            if (!p || !isfinite(*p)) {
                bad = 1;
                continue;
            }

            float e = expected[i * cols + j];
            float diff = fabsf(*p - e);
            if (diff > max_abs) max_abs = diff;

            if (!almost_equal(*p, e, abs_tol, rel_tol)) {
                bad = 1;
                printf("       Error en [%d,%d]: esperado = % .9f, obtenido = % .9f, |error| = %.9g\n",
                       i, j, e, *p, diff);
            }
        }
    }

    if (bad) {
        snprintf(reason, sizeof(reason), "La matriz no coincide. Error absoluto maximo = %.9g.", max_abs);
        result_fail(name, reason);
        print_matrix("obtenida", actual);
        return 0;
    }

    result_ok(name);
    return 1;
}

static struct matrix *new_matrix(int rows, int cols, const float *data)
{
    struct matrix *m = NULL;
    crearMatriz(&m, rows, cols);
    if (!m) return NULL;

    if (data) {
        /* inicializarMatriz recibe float* aunque solo necesite leer. */
        inicializarMatriz(m, (float *)data);
    }
    return m;
}

static void free_matrix(struct matrix **m)
{
    if (m && *m) {
        eliminarMatriz(m);
    }
}

static struct layer *new_layer_from_arrays(int rows_w,
                                            int cols_w,
                                            const float *w,
                                            int rows_b,
                                            int cols_b,
                                            const float *b,
                                            activacion act)
{
    struct matrix *mw = new_matrix(rows_w, cols_w, w);
    struct matrix *mb = new_matrix(rows_b, cols_b, b);
    struct layer *layer = NULL;

    if (!mw || !mb) {
        free_matrix(&mw);
        free_matrix(&mb);
        return NULL;
    }

    int rc = crearLayer(mw, mb, act, &layer);
    free_matrix(&mw);
    free_matrix(&mb);

    if (rc != 0) {
        if (layer) eliminarLayer(&layer);
        return NULL;
    }
    return layer;
}

static void free_layer(struct layer **layer)
{
    if (layer && *layer) {
        eliminarLayer(layer);
    }
}

static struct network *new_network(void)
{
    struct network *red = NULL;
    int rc = crearRed(&red);
    if (rc != 0 || !red) {
        if (red) eliminarRed(&red);
        return NULL;
    }
    return red;
}

static int add_layer_checked(struct network *red, struct layer **layer)
{
    if (!red || !layer || !*layer) return 0;

    struct layer *candidate = *layer;
    int rc = añadirCapa(candidate, red);

    if (rc == 0) {
        /* La red pasa a ser propietaria. */
        *layer = NULL;
        return 1;
    }

    return 0;
}

static void free_network(struct network **red)
{
    if (red && *red) {
        eliminarRed(red);
    }
}

static int check_return_code(const char *name, int actual, int expected)
{
    char reason[160];
    if (actual == expected) {
        result_ok(name);
        return 1;
    }

    snprintf(reason, sizeof(reason),
             "Codigo devuelto incorrecto. Esperado = %d, obtenido = %d.",
             expected, actual);
    result_fail(name, reason);
    return 0;
}

/* ============================================================
 * 1. CREACION / PROPIEDAD / ANADIR CAPAS
 * ============================================================ */

static void test_crear_red(void)
{
    section("1. CREAR RED Y ESTADO INICIAL");

    struct network *red = NULL;
    int rc = crearRed(&red);

    check_return_code("crearRed devuelve 0", rc, 0);

    if (!red) {
        result_fail("crearRed devuelve una red no NULL", "red == NULL despues de crearRed().");
        return;
    }
    result_ok("crearRed devuelve una red no NULL");

    if (red->numeroCapas == 0) {
        result_ok("red nueva tiene 0 capas");
    } else {
        char reason[160];
        snprintf(reason, sizeof(reason), "numeroCapas esperado = 0, obtenido = %d.", red->numeroCapas);
        result_fail("red nueva tiene 0 capas", reason);
    }

    free_network(&red);

    if (!red) {
        result_ok("eliminarRed deja el puntero a NULL");
    } else {
        result_fail("eliminarRed deja el puntero a NULL",
                    "El puntero sigue siendo no NULL despues de eliminarRed().");
    }
}

static void test_crear_layer_deep_copy(void)
{
    section("2. COPIA PROFUNDA DE crearLayer (base para ownership de network)");

    float wdata[] = {1, 2, 3, 4};
    float bdata[] = {5, 6};
    struct matrix *w = new_matrix(2, 2, wdata);
    struct matrix *b = new_matrix(2, 1, bdata);
    struct layer *layer = NULL;

    int rc = crearLayer(w, b, TANH, &layer);
    check_return_code("crearLayer devuelve 0", rc, 0);

    if (!layer) {
        result_fail("crearLayer devuelve layer no NULL", "layer == NULL.");
        free_matrix(&w);
        free_matrix(&b);
        return;
    }
    result_ok("crearLayer devuelve layer no NULL");

    /* Cambiamos las matrices originales: la capa no deberia cambiar. */
    modificarPos(w, 0, 0, 999.0f);
    modificarPos(b, 0, 0, -999.0f);

    float expected_w[] = {1, 2, 3, 4};
    float expected_b[] = {5, 6};
    check_matrix("crearLayer hace copia profunda de pesos", layer->pesos, 2, 2,
                 expected_w, ABS_TOL, REL_TOL);
    check_matrix("crearLayer hace copia profunda de bias", layer->bias, 2, 1,
                 expected_b, ABS_TOL, REL_TOL);

    if (layer->nEntradas == 2 && layer->nSalidas == 2) {
        result_ok("crearLayer infiere nEntradas/nSalidas correctamente");
    } else {
        char reason[200];
        snprintf(reason, sizeof(reason),
                 "Esperado nEntradas=2, nSalidas=2; obtenido nEntradas=%d, nSalidas=%d.",
                 layer->nEntradas, layer->nSalidas);
        result_fail("crearLayer infiere nEntradas/nSalidas correctamente", reason);
    }

    free_layer(&layer);
    free_matrix(&w);
    free_matrix(&b);
}

static void test_anadir_capas_y_validacion(void)
{
    section("3. añadirCapa: orden, dimensiones y rechazo de incompatibles");

    struct network *red = new_network();
    if (!red) {
        result_fail("precondicion crearRed para añadirCapa", "No se pudo crear la red.");
        return;
    }

    float w1[] = {1, 0, 0, 1};
    float b1[] = {0, 0};
    struct layer *l1 = new_layer_from_arrays(2, 2, w1, 2, 1, b1, RELU);

    if (!l1 || !add_layer_checked(red, &l1)) {
        result_fail("añadirCapa acepta la primera capa valida", "No se pudo añadir la capa.");
        free_layer(&l1);
        free_network(&red);
        return;
    }
    result_ok("añadirCapa acepta la primera capa valida");

    float w2[] = {2, 3};
    float b2[] = {1};
    struct layer *l2 = new_layer_from_arrays(1, 2, w2, 1, 1, b2, RELU);
    struct layer *l2_original = l2;

    if (!l2) {
        result_fail("crear segunda capa valida", "No se pudo crear l2.");
        free_network(&red);
        return;
    }

    int rc = añadirCapa(l2, red);
    check_return_code("añadirCapa acepta segunda capa compatible", rc, 0);

    if (rc == 0) {
        l2 = NULL; /* ownership transferido */
    } else {
        free_layer(&l2);
    }

    if (red->numeroCapas == 2) {
        result_ok("la red contiene exactamente 2 capas");
    } else {
        char reason[160];
        snprintf(reason, sizeof(reason), "Esperado 2 capas, obtenido %d.", red->numeroCapas);
        result_fail("la red contiene exactamente 2 capas", reason);
    }

    if (red->numeroCapas >= 2 && red->capas[0] && red->capas[1] == l2_original) {
        result_ok("añadirCapa conserva el puntero/ownership de la capa");
    } else if (red->numeroCapas >= 2) {
        result_fail("añadirCapa conserva el puntero/ownership de la capa",
                    "La segunda capa de la red no coincide con el puntero pasado.");
    }

    /* Capa incompatible: la anterior termina en 2 salidas, esta espera 3 entradas. */
    float w_bad[] = {1, 2, 3};
    float b_bad[] = {0};
    struct layer *bad = new_layer_from_arrays(1, 3, w_bad, 1, 1, b_bad, RELU);
    int capas_antes = red->numeroCapas;

    if (!bad) {
        result_fail("crear capa incompatible para el test", "No se pudo crear la capa candidata.");
    } else {
        int bad_rc = añadirCapa(bad, red);

        if (bad_rc == 1) {
            result_ok("añadirCapa rechaza una capa dimensionalmente incompatible");
            if (red->numeroCapas == capas_antes) {
                result_ok("añadirCapa no modifica la red al rechazar una capa");
            } else {
                result_fail("añadirCapa no modifica la red al rechazar una capa",
                            "numeroCapas cambio pese a haber devuelto error.");
            }
            free_layer(&bad);
        } else {
            result_fail("añadirCapa rechaza una capa dimensionalmente incompatible",
                        "Se esperaba codigo 1 para una capa que no puede multiplicarse con la salida anterior.");
            /* Si la implementacion la anadio pese al error esperado, la red se encarga de ella. */
            if (red->numeroCapas == capas_antes + 1) {
                printf("       La implementacion parece haber transferido ownership pese al contrato esperado.\n");
                bad = NULL;
            } else {
                free_layer(&bad);
            }
        }
    }

    free_network(&red);
}

/* ============================================================
 * 4. FORWARD Y ACTIVACIONES A TRAVES DE network
 * ============================================================ */

static void test_forward_activation_relu(void)
{
    section("4. forwardRed + RELU");

    struct network *red = new_network();
    float w[] = {1, 0, 0, 1, 0, 0};
    float b[] = {0, 0, 0};
    /* 3x2 identidad embebida: z = [x0, x1, 0] */
    struct layer *layer = new_layer_from_arrays(3, 2, w, 3, 1, b, RELU);
    float xdata[] = {-2, 0};
    struct matrix *x = new_matrix(2, 1, xdata);

    if (!red || !layer || !x) {
        result_fail("precondiciones RELU", "No se pudieron construir red/capa/entrada.");
        free_layer(&layer);
        free_matrix(&x);
        free_network(&red);
        return;
    }

    if (!add_layer_checked(red, &layer)) {
        result_fail("añadir capa RELU", "añadirCapa devolvio error inesperado.");
        free_layer(&layer);
        free_matrix(&x);
        free_network(&red);
        return;
    }

    struct matrix *out = forwardRed(red, x);
    float expected[] = {0, 0, 0};
    check_matrix("forwardRed RELU", out, 3, 1, expected, ABS_TOL, REL_TOL);

    /* z no es la salida activada: los tres z esperados son [-2,0,0]. */
    float expected_z[] = {-2, 0, 0};
    if (red->cache && red->cache[0]) {
        check_matrix("cache[0]->z RELU", red->cache[0]->z, 3, 1,
                     expected_z, ABS_TOL, REL_TOL);
        check_matrix("cache[0]->a RELU", red->cache[0]->a, 3, 1,
                     expected, ABS_TOL, REL_TOL);
    } else {
        result_fail("forwardRed crea cache para la capa RELU", "red->cache[0] es NULL.");
    }

    /* El resultado devuelto debe poder liberarse sin destruir el cache. */
    free_matrix(&out);
    if (red->cache && red->cache[0] && red->cache[0]->a) {
        result_ok("liberar el resultado devuelto no elimina el cache de forward");
    } else {
        result_fail("liberar el resultado devuelto no elimina el cache de forward",
                    "El cache dejo de existir despues de liberar el resultado.");
    }

    free_matrix(&x);
    free_network(&red);
}

static void test_forward_activation_sigmoid(void)
{
    section("5. forwardRed + SIGMOID");

    struct network *red = new_network();
    float w[] = {1, 0, 0, 1, 0, 0};
    float b[] = {0, 0, 0};
    struct layer *layer = new_layer_from_arrays(3, 2, w, 3, 1, b, SIGMOID);
    float xdata[] = {0, 2};
    struct matrix *x = new_matrix(2, 1, xdata);

    if (!red || !layer || !x || !add_layer_checked(red, &layer)) {
        result_fail("precondiciones SIGMOID", "No se pudo montar la red.");
        free_layer(&layer);
        free_matrix(&x);
        free_network(&red);
        return;
    }

    struct matrix *out = forwardRed(red, x);
    float expected[] = {
        0.5f,
        0.880797077f,
        0.5f
    };
    check_matrix("forwardRed SIGMOID", out, 3, 1, expected, 2e-5f, 2e-5f);

    free_matrix(&out);
    free_matrix(&x);
    free_network(&red);
}

static void test_forward_activation_tanh(void)
{
    section("6. forwardRed + TANH");

    struct network *red = new_network();
    float w[] = {1, 0, 0, 1, 0, 0};
    float b[] = {0, 0, 0};
    struct layer *layer = new_layer_from_arrays(3, 2, w, 3, 1, b, TANH);
    float xdata[] = {-1, 2};
    struct matrix *x = new_matrix(2, 1, xdata);

    if (!red || !layer || !x || !add_layer_checked(red, &layer)) {
        result_fail("precondiciones TANH", "No se pudo montar la red.");
        free_layer(&layer);
        free_matrix(&x);
        free_network(&red);
        return;
    }

    struct matrix *out = forwardRed(red, x);
    float expected[] = {
        -0.761594156f,
         0.964027580f,
         0.0f
    };
    check_matrix("forwardRed TANH", out, 3, 1, expected, 2e-5f, 2e-5f);

    free_matrix(&out);
    free_matrix(&x);
    free_network(&red);
}

static struct network *build_two_layer_relu_network(void)
{
    struct network *red = new_network();
    if (!red) return NULL;

    float w1[] = {
         1.0f,  0.5f,
        -0.5f, -1.0f
    };
    float b1[] = {0.0f, 2.0f};
    float w2[] = {2.0f, -1.0f};
    float b2[] = {0.5f};

    struct layer *l1 = new_layer_from_arrays(2, 2, w1, 2, 1, b1, RELU);
    struct layer *l2 = new_layer_from_arrays(1, 2, w2, 1, 1, b2, RELU);

    if (!l1 || !l2) {
        free_layer(&l1);
        free_layer(&l2);
        free_network(&red);
        return NULL;
    }

    if (!add_layer_checked(red, &l1) || !add_layer_checked(red, &l2)) {
        free_layer(&l1);
        free_layer(&l2);
        free_network(&red);
        return NULL;
    }

    return red;
}

static void test_forward_two_layers_and_cache(void)
{
    section("7. forwardRed con varias capas + propagacion + caches");

    struct network *red = build_two_layer_relu_network();
    float xdata[] = {2.0f, -1.0f};
    struct matrix *x = new_matrix(2, 1, xdata);

    if (!red || !x) {
        result_fail("precondiciones forward multicapa", "No se pudo montar la red.");
        free_matrix(&x);
        free_network(&red);
        return;
    }

    struct matrix *out = forwardRed(red, x);
    float expected_out[] = {1.5f};
    check_matrix("forwardRed multicapa", out, 1, 1, expected_out, ABS_TOL, REL_TOL);

    if (!red->cache || !red->cache[0] || !red->cache[1]) {
        result_fail("forwardRed crea cache para todas las capas", "Falta al menos un cache.");
    } else {
        result_ok("forwardRed crea cache para todas las capas");

        float z1[] = {1.5f, 2.0f};
        float a1[] = {1.5f, 2.0f};
        float z2[] = {1.5f};
        float a2[] = {1.5f};

        check_matrix("cache capa 0: z", red->cache[0]->z, 2, 1, z1, ABS_TOL, REL_TOL);
        check_matrix("cache capa 0: a", red->cache[0]->a, 2, 1, a1, ABS_TOL, REL_TOL);
        check_matrix("cache capa 1: z", red->cache[1]->z, 1, 1, z2, ABS_TOL, REL_TOL);
        check_matrix("cache capa 1: a", red->cache[1]->a, 1, 1, a2, ABS_TOL, REL_TOL);
    }

    /* Segundo forward con otra entrada: detecta cache obsoleto. */
    float xdata2[] = {1.0f, 2.0f};
    struct matrix *x2 = new_matrix(2, 1, xdata2);
    if (!x2) {
        result_fail("crear segunda entrada para cache", "No se pudo crear x2.");
    } else {
        struct matrix *out2 = forwardRed(red, x2);
        float expected_out2[] = {4.5f};
        check_matrix("forwardRed segundo paso (cache actualizado)", out2, 1, 1,
                     expected_out2, ABS_TOL, REL_TOL);
        free_matrix(&out2);
        free_matrix(&x2);
    }

    free_matrix(&out);
    free_matrix(&x);
    free_network(&red);
}

/* ============================================================
 * 8. BACKPROP EXACTO
 * ============================================================ */

static void test_backprop_exact(void)
{
    section("8. backpropRed: gradientes exactos en red de 2 capas");

    struct network *red = build_two_layer_relu_network();
    float xdata[] = {2.0f, -1.0f};
    float target_data[] = {0.5f};
    struct matrix *x = new_matrix(2, 1, xdata);
    struct matrix *target = new_matrix(1, 1, target_data);

    if (!red || !x || !target) {
        result_fail("precondiciones backprop exacto", "No se pudo montar la red/entrada/objetivo.");
        free_matrix(&x);
        free_matrix(&target);
        free_network(&red);
        return;
    }

    struct matrix *out = forwardRed(red, x);
    float expected_out[] = {1.5f};
    check_matrix("salida previa al backprop", out, 1, 1, expected_out, ABS_TOL, REL_TOL);

    /* n = 1, por lo que d(1/2*(y-t)^2)/dy = y-t = 1. */
    int rc = backpropRed(red, *out, *target);
    check_return_code("backpropRed devuelve 0", rc, 0);

    if (rc == 0 && red->numeroCapas >= 2 &&
        red->capas[0] && red->capas[1] &&
        red->capas[0]->gradientes && red->capas[1]->gradientes) {
        result_ok("backpropRed crea gradientes en todas las capas");
    } else {
        result_fail("backpropRed crea gradientes en todas las capas",
                    "Falta gradientes en una o mas capas.");
        free_matrix(&out);
        free_matrix(&target);
        free_matrix(&x);
        free_network(&red);
        return;
    }

    /* Capa 2 */
    float exp_dw2[] = {1.5f, 2.0f};
    float exp_db2[] = {1.0f};
    float exp_dx2[] = {2.0f, -1.0f};
    check_matrix("capa 2 dL_dw", red->capas[1]->gradientes->dL_dw,
                 1, 2, exp_dw2, ABS_TOL, REL_TOL);
    check_matrix("capa 2 dL_db", red->capas[1]->gradientes->dL_db,
                 1, 1, exp_db2, ABS_TOL, REL_TOL);
    check_matrix("capa 2 dL_dx", red->capas[1]->gradientes->dL_dx,
                 2, 1, exp_dx2, ABS_TOL, REL_TOL);

    /* Capa 1 */
    float exp_dw1[] = {
         4.0f, -2.0f,
        -2.0f,  1.0f
    };
    float exp_db1[] = {2.0f, -1.0f};
    check_matrix("capa 1 dL_dw", red->capas[0]->gradientes->dL_dw,
                 2, 2, exp_dw1, ABS_TOL, REL_TOL);
    check_matrix("capa 1 dL_db", red->capas[0]->gradientes->dL_db,
                 2, 1, exp_db1, ABS_TOL, REL_TOL);


    /* El error de perdida esperado tambien sirve como comprobacion adicional. */
    float loss = mse(*out, *target);
    if (almost_equal(loss, 0.5f, 2e-5f, 2e-5f)) {
        result_ok("mse del caso exacto = 0.5");
    } else {
        char reason[200];
        snprintf(reason, sizeof(reason), "Esperado 0.5, obtenido %.9f.", loss);
        result_fail("mse del caso exacto = 0.5", reason);
    }

    free_matrix(&out);
    free_matrix(&target);
    free_matrix(&x);
    free_network(&red);
}

static void test_backprop_refresh(void)
{
    section("9. backpropRed repetido: debe actualizar caches y gradientes");

    struct network *red = build_two_layer_relu_network();
    float xdata[] = {2.0f, -1.0f};
    float target_data[] = {0.5f};
    struct matrix *x = new_matrix(2, 1, xdata);
    struct matrix *target = new_matrix(1, 1, target_data);

    if (!red || !x || !target) {
        result_fail("precondiciones backprop repetido", "No se pudo montar el caso.");
        free_matrix(&x);
        free_matrix(&target);
        free_network(&red);
        return;
    }

    struct matrix *out = forwardRed(red, x);
    if (out) {
        backpropRed(red, *out, *target);
    }
    free_matrix(&out);

    /* Segunda entrada: hidden = [2,0], salida = 4.5, target = 1 => delta = 3.5. */
    float xdata2[] = {1.0f, 2.0f};
    float target_data2[] = {1.0f};
    struct matrix *x2 = new_matrix(2, 1, xdata2);
    struct matrix *target2 = new_matrix(1, 1, target_data2);

    if (!x2 || !target2) {
        result_fail("crear segunda entrada/target", "No se pudieron crear matrices.");
        free_matrix(&x2);
        free_matrix(&target2);
        free_matrix(&x);
        free_matrix(&target);
        free_network(&red);
        return;
    }

    struct matrix *out2 = forwardRed(red, x2);
    float expected_out2[] = {4.5f};
    check_matrix("salida del segundo paso", out2, 1, 1, expected_out2, ABS_TOL, REL_TOL);

    int rc = backpropRed(red, *out2, *target2);
    check_return_code("backpropRed segundo paso devuelve 0", rc, 0);

    float exp_dw2[] = {7.0f, 0.0f};
    float exp_db2[] = {3.5f};
    float exp_dx2[] = {7.0f, -3.5f};
    float exp_dw1[] = {
         7.0f, 14.0f,
         0.0f,  0.0f
    };
    float exp_db1[] = {7.0f, 0.0f};

    check_matrix("segundo paso capa 2 dL_dw", red->capas[1]->gradientes->dL_dw,
                 1, 2, exp_dw2, ABS_TOL, REL_TOL);
    check_matrix("segundo paso capa 2 dL_db", red->capas[1]->gradientes->dL_db,
                 1, 1, exp_db2, ABS_TOL, REL_TOL);
    check_matrix("segundo paso capa 2 dL_dx", red->capas[1]->gradientes->dL_dx,
                 2, 1, exp_dx2, ABS_TOL, REL_TOL);
    check_matrix("segundo paso capa 1 dL_dw", red->capas[0]->gradientes->dL_dw,
                 2, 2, exp_dw1, ABS_TOL, REL_TOL);
    check_matrix("segundo paso capa 1 dL_db", red->capas[0]->gradientes->dL_db,
                 2, 1, exp_db1, ABS_TOL, REL_TOL);

    free_matrix(&out2);
    free_matrix(&x2);
    free_matrix(&target2);
    free_matrix(&x);
    free_matrix(&target);
    free_network(&red);
}

/* ============================================================
 * 10. GRADIENT CHECKING NUMERICO
 * ============================================================ */

static struct network *build_gradient_check_network(void)
{
    struct network *red = new_network();
    if (!red) return NULL;

    /* Hidden sigmoid -> output tanh. Todo esta lejos de los puntos
       no derivables/saturaciones fuertes. */
    float w1[] = {
         0.4f, -0.2f,
         0.3f,  0.5f
    };
    float b1[] = {0.1f, -0.1f};
    float w2[] = {0.8f, -0.6f};
    float b2[] = {0.05f};

    struct layer *l1 = new_layer_from_arrays(2, 2, w1, 2, 1, b1, SIGMOID);
    struct layer *l2 = new_layer_from_arrays(1, 2, w2, 1, 1, b2, TANH);

    if (!l1 || !l2 || !add_layer_checked(red, &l1) || !add_layer_checked(red, &l2)) {
        free_layer(&l1);
        free_layer(&l2);
        free_network(&red);
        return NULL;
    }

    return red;
}

static int finite_difference_param(struct network *red,
                                   struct matrix *x,
                                   struct matrix *target,
                                   float *param,
                                   float eps,
                                   float *numeric_gradient,
                                   float *loss_plus,
                                   float *loss_minus)
{
    if (!red || !x || !target || !param || !numeric_gradient) return 0;

    float original = *param;

    *param = original + eps;
    struct matrix *plus = forwardRed(red, x);
    if (!plus) {
        *param = original;
        return 0;
    }
    *loss_plus = mse(*plus, *target);
    free_matrix(&plus);

    *param = original - eps;
    struct matrix *minus = forwardRed(red, x);
    if (!minus) {
        *param = original;
        return 0;
    }
    *loss_minus = mse(*minus, *target);
    free_matrix(&minus);

    *param = original;

    *numeric_gradient = (*loss_plus - *loss_minus) / (2.0f * eps);
    return isfinite(*numeric_gradient);
}

static void compare_one_gradient(const char *name,
                                 float analytical,
                                 float numerical,
                                 float loss_plus,
                                 float loss_minus)
{
    float diff = fabsf(analytical - numerical);
    float limit = GRAD_ABS_TOL + GRAD_REL_TOL * fabsf(numerical);

    char reason[320];
    if (isfinite(analytical) && isfinite(numerical) && diff <= limit) {
        result_ok(name);
    } else {
        snprintf(reason, sizeof(reason),
                 "Gradiente analitico = % .9f, numerico = % .9f, |error| = %.9g, limite = %.9g. "
                 "L(+eps) = %.9f, L(-eps) = %.9f",
                 analytical, numerical, diff, limit, loss_plus, loss_minus);
        result_fail(name, reason);
    }
}

static void test_gradient_checking(void)
{
    section("10. GRADIENT CHECKING: dL/dparam vs diferencia finita");

    struct network *red = build_gradient_check_network();
    float xdata[] = {0.7f, -1.1f};
    float target_data[] = {0.2f};
    struct matrix *x = new_matrix(2, 1, xdata);
    struct matrix *target = new_matrix(1, 1, target_data);

    if (!red || !x || !target) {
        result_fail("precondiciones gradient checking", "No se pudo montar el caso.");
        free_matrix(&x);
        free_matrix(&target);
        free_network(&red);
        return;
    }

    struct matrix *out = forwardRed(red, x);
    if (!out) {
        result_fail("forward previo al gradient checking", "forwardRed devolvio NULL.");
        free_matrix(&x);
        free_matrix(&target);
        free_network(&red);
        return;
    }

    float base_loss = mse(*out, *target);
    printf("       Perdida base = %.9f\n", base_loss);

    int rc = backpropRed(red, *out, *target);
    check_return_code("backpropRed para gradient checking devuelve 0", rc, 0);
    free_matrix(&out);

    if (rc != 0) {
        free_matrix(&x);
        free_matrix(&target);
        free_network(&red);
        return;
    }

    /* Recorremos todos los pesos y biases de todas las capas. */
    for (int li = 0; li < red->numeroCapas; ++li) {
        struct layer *layer = red->capas[li];
        if (!layer || !layer->gradientes) {
            char name[128];
            snprintf(name, sizeof(name), "gradient checking capa %d tiene gradientes", li);
            result_fail(name, "gradientes == NULL.");
            continue;
        }

        for (int i = 0; i < layer->pesos->fil; ++i) {
            for (int j = 0; j < layer->pesos->col; ++j) {
                float *param = accederPos(layer->pesos, i, j);
                float *g = accederPos(layer->gradientes->dL_dw, i, j);
                float numerical = 0.0f;
                float lp = 0.0f, lm = 0.0f;

                int ok = finite_difference_param(red, x, target, param, FD_EPS,
                                                 &numerical, &lp, &lm);
                char name[180];
                snprintf(name, sizeof(name),
                         "gradcheck capa %d dL/dw[%d,%d]", li, i, j);

                if (!ok || !g) {
                    result_fail(name, "No se pudo calcular la derivada numerica o falta el gradiente analitico.");
                } else {
                    compare_one_gradient(name, *g, numerical, lp, lm);
                }
            }
        }

        for (int i = 0; i < layer->bias->fil; ++i) {
            for (int j = 0; j < layer->bias->col; ++j) {
                float *param = accederPos(layer->bias, i, j);
                float *g = accederPos(layer->gradientes->dL_db, i, j);
                float numerical = 0.0f;
                float lp = 0.0f, lm = 0.0f;

                int ok = finite_difference_param(red, x, target, param, FD_EPS,
                                                 &numerical, &lp, &lm);
                char name[180];
                snprintf(name, sizeof(name),
                         "gradcheck capa %d dL/db[%d,%d]", li, i, j);

                if (!ok || !g) {
                    result_fail(name, "No se pudo calcular la derivada numerica o falta el gradiente analitico.");
                } else {
                    compare_one_gradient(name, *g, numerical, lp, lm);
                }
            }
        }
    }

    /* Tras modificar parametros y restaurarlos, el forward+backprop final
       debe seguir produciendo gradientes coherentes. */
    struct matrix *final_out = forwardRed(red, x);
    if (final_out) {
        int rc2 = backpropRed(red, *final_out, *target);
        check_return_code("backprop final despues de restaurar parametros", rc2, 0);
        free_matrix(&final_out);
    } else {
        result_fail("forward final despues de restaurar parametros", "forwardRed devolvio NULL.");
    }

    free_matrix(&x);
    free_matrix(&target);
    free_network(&red);
}

/* ============================================================
 * 11. TESTS DE INVARIANTES Y CASOS LIMITE DE network
 * ============================================================ */

static void test_single_layer_non_square(void)
{
    section("11. CAPA NO CUADRADA: comprueba que no se confunden filas/columnas");

    struct network *red = new_network();
    /* W = 2 x 3, bias = 2 x 1. */
    float w[] = {
        1, 2, 3,
        4, 5, 6
    };
    float b[] = {10, -10};
    float xdata[] = {1, 2, 3};
    struct layer *layer = new_layer_from_arrays(2, 3, w, 2, 1, b, RELU);
    struct matrix *x = new_matrix(3, 1, xdata);

    if (!red || !layer || !x || !add_layer_checked(red, &layer)) {
        result_fail("precondiciones capa no cuadrada", "No se pudo montar el caso.");
        free_layer(&layer);
        free_matrix(&x);
        free_network(&red);
        return;
    }

    /* z0 = 1+4+9+10 = 24; z1 = 4+10+18-10=22. */
    struct matrix *out = forwardRed(red, x);
    float expected[] = {24, 22};
    check_matrix("forwardRed capa 2x3", out, 2, 1, expected, ABS_TOL, REL_TOL);

    free_matrix(&out);
    free_matrix(&x);
    free_network(&red);
}

static void test_free_after_forward_and_backprop(void)
{
    section("12. CICLO COMPLETO DE MEMORIA: forward -> backprop -> eliminarRed");

    struct network *red = build_gradient_check_network();
    float xdata[] = {0.3f, -0.4f};
    float target_data[] = {0.1f};
    struct matrix *x = new_matrix(2, 1, xdata);
    struct matrix *target = new_matrix(1, 1, target_data);

    if (!red || !x || !target) {
        result_fail("precondiciones ciclo de memoria", "No se pudo montar el caso.");
        free_matrix(&x);
        free_matrix(&target);
        free_network(&red);
        return;
    }

    struct matrix *out = forwardRed(red, x);
    if (!out) {
        result_fail("forward del ciclo de memoria", "forwardRed devolvio NULL.");
    } else {
        int rc = backpropRed(red, *out, *target);
        check_return_code("backprop del ciclo de memoria", rc, 0);
        free_matrix(&out);
        result_ok("resultado de forward liberado antes de eliminarRed");
    }

    free_matrix(&x);
    free_matrix(&target);
    free_network(&red);

    if (!red) {
        result_ok("eliminarRed deja NULL tras ciclo completo");
    } else {
        result_fail("eliminarRed deja NULL tras ciclo completo",
                    "El puntero de red no quedo NULL.");
    }
}

static void print_summary(void)
{
    printf("\n");
    line();
    printf("RESUMEN FINAL\n");
    line();
    printf("Tests totales : %d\n", total_tests);
    printf("Tests PASS    : %d\n", passed_tests);
    printf("Tests FAIL    : %d\n", failed_tests);

    if (failed_tests == 0) {
        printf("ESTADO        : TODO PASS\n");
    } else {
        printf("ESTADO        : HAY ERRORES\n");
        printf("Revisa los bloques [FAIL] anteriores; cada uno imprime los valores\n");
        printf("esperado/obtenido cuando la comprobacion es numerica.\n");
    }
    line();
}

int main(void)
{
    printf("============================================================\n");
    printf(" TEST EXHAUSTIVO DE LA LIBRERIA NETWORK\n");
    printf("============================================================\n");
    printf("Tolerancias normales : abs %.1e / rel %.1e\n", ABS_TOL, REL_TOL);
    printf("Tolerancias gradiente: abs %.1e / rel %.1e\n", GRAD_ABS_TOL, GRAD_REL_TOL);
    printf("Finite differences   : eps %.1e\n", FD_EPS);

    test_crear_red();
    test_crear_layer_deep_copy();
    test_anadir_capas_y_validacion();
    test_forward_activation_relu();
    test_forward_activation_sigmoid();
    test_forward_activation_tanh();
    test_forward_two_layers_and_cache();
    test_backprop_exact();
    test_backprop_refresh();
    test_gradient_checking();
    test_single_layer_non_square();
    test_free_after_forward_and_backprop();

    print_summary();

    return failed_tests == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
