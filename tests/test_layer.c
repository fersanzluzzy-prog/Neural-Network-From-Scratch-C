#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "layer.h"
#include "matrix.h"
#include "activation.h"

/*
 * ============================================================================
 * TEST LAYER
 * ============================================================================
 *
 * Funciones probadas:
 *
 *   - crearLayer()
 *   - eliminarLayer()
 *   - forward()
 *
 * Funciones relacionadas con cache:
 *
 *   - inicializarCache()
 *   - guardarCacheLayerA()
 *   - guardarCacheLayerZ()
 *   - eliminarCacheLayer()
 *
 * NO se utiliza:
 *
 *   - guardarCacheLayer()
 *
 * porque está declarada en layer.h pero no está implementada.
 *
 *
 * MODELO DE UNA CAPA
 * ------------------
 *
 * La capa realiza:
 *
 *      Z = W * A + b
 *      A = f(Z)
 *
 * donde:
 *
 *      W -> pesos
 *      b -> bias
 *      A -> entrada
 *      Z -> preactivación
 *      A_salida -> activación
 *
 *
 * Para una capa:
 *
 *      nEntradas = 2
 *      nSalidas  = 3
 *
 * esperamos:
 *
 *      W = 3 x 2
 *      b = 3 x 1
 *      entrada = 2 x 1
 *      salida  = 3 x 1
 *
 *
 * OBJETIVOS
 * ---------
 *
 * Se comprueba:
 *
 *   1. Creación correcta de la estructura layer.
 *   2. Asociación correcta de pesos.
 *   3. Asociación/copia correcta de bias.
 *   4. nEntradas correcto.
 *   5. nSalidas correcto.
 *   6. Activación correcta.
 *   7. Gradientes inicialmente NULL.
 *   8. Forward con SIGMOID.
 *   9. Forward con TANH.
 *  10. Forward con RELU.
 *  11. Valores positivos y negativos.
 *  12. Entrada cero.
 *  13. Dimensiones de entrada/salida.
 *  14. Cache generado por forward.
 *  15. Cache contiene A y Z.
 *  16. Z coincide con W*A+b.
 *  17. A coincide con f(Z).
 *  18. Sustitución de valores de entrada.
 *  19. Eliminación de layer.
 *
 * Las comprobaciones muestran:
 *
 *   - qué test ha fallado
 *   - qué operación se estaba realizando
 *   - dimensiones
 *   - valores esperados
 *   - valores obtenidos
 *   - error absoluto
 *   - información del cache
 *   - código de retorno cuando existe
 * ========================================================================== */


#define TOL_ABS 1e-5f
#define TOL_REL 1e-5f

static int tests_totales = 0;
static int tests_pasados = 0;
static int tests_fallados = 0;


/* ============================================================================
 * UTILIDADES
 * ========================================================================== */

static int floatIguales(float a, float b)
{
    float error = fabsf(a - b);
    float escala = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));

    return error <= TOL_ABS ||
           error / escala <= TOL_REL;
}


static void separador(void)
{
    printf("-------------------------------------------------------------------------------\n");
}


static void imprimirMatriz(
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

    printf(
        "%s [%d x %d]\n",
        nombre,
        m->fil,
        m->col
    );

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
            printf(
                "% .8f",
                m->datos[i * m->col + j]
            );

            if (j < m->col - 1)
                printf(", ");
        }

        printf(" ]\n");
    }
}


static void imprimirLayer(
    const char *nombre,
    const struct layer *layer)
{
    if (layer == NULL)
    {
        printf("%s = NULL\n", nombre);
        return;
    }

    printf("%s\n", nombre);

    printf(
        "  dirección       : %p\n",
        (void *)layer
    );

    printf(
        "  nEntradas       : %d\n",
        layer->nEntradas
    );

    printf(
        "  nSalidas        : %d\n",
        layer->nSalidas
    );

    printf(
        "  pesos           : %p\n",
        (void *)layer->pesos
    );

    printf(
        "  bias            : %p\n",
        (void *)layer->bias
    );

    printf(
        "  gradientes      : %p\n",
        (void *)layer->gradientes
    );

    printf(
        "  activacion enum : %d\n",
        (int)layer->activacion
    );

    if (layer->pesos != NULL)
        imprimirMatriz("  W", layer->pesos);

    if (layer->bias != NULL)
        imprimirMatriz("  b", layer->bias);
}


static void registrarResultado(
    const char *nombre,
    int correcto)
{
    tests_totales++;

    if (correcto)
    {
        tests_pasados++;
        printf("[PASS] %s\n", nombre);
    }
    else
    {
        tests_fallados++;
        printf("[FAIL] %s\n", nombre);
    }

    printf("\n");
}


static int comprobarMatriz(
    const char *nombre,
    const struct matrix *obtenida,
    const float *esperada,
    int filas,
    int columnas)
{
    int i;
    int j;
    int errores = 0;

    if (obtenida == NULL)
    {
        printf(
            "  [ERROR] %s == NULL.\n",
            nombre
        );

        return 0;
    }

    if (obtenida->fil != filas ||
        obtenida->col != columnas)
    {
        printf(
            "  [ERROR] Dimensiones incorrectas para %s.\n"
            "          Esperadas: %d x %d\n"
            "          Obtenidas : %d x %d\n",
            nombre,
            filas,
            columnas,
            obtenida->fil,
            obtenida->col
        );

        return 0;
    }

    if (obtenida->datos == NULL)
    {
        printf(
            "  [ERROR] %s->datos == NULL.\n",
            nombre
        );

        return 0;
    }

    for (i = 0; i < filas; i++)
    {
        for (j = 0; j < columnas; j++)
        {
            int indice = i * columnas + j;

            float esperado = esperada[indice];
            float obtenido = obtenida->datos[indice];

            if (!floatIguales(obtenido, esperado))
            {
                printf(
                    "  [ERROR] %s[%d][%d]\n"
                    "          Esperado : %.10f\n"
                    "          Obtenido : %.10f\n"
                    "          Error abs: %.10f\n",
                    nombre,
                    i,
                    j,
                    esperado,
                    obtenido,
                    fabsf(obtenido - esperado)
                );

                errores++;
            }
        }
    }

    if (errores > 0)
    {
        printf(
            "  [ERROR] %d elemento(s) incorrecto(s) en %s.\n",
            errores,
            nombre
        );

        return 0;
    }

    return 1;
}


static int comprobarPuntero(
    const char *nombre,
    const void *puntero)
{
    if (puntero == NULL)
    {
        printf(
            "  [ERROR] %s == NULL.\n",
            nombre
        );

        return 0;
    }

    printf(
        "  [OK] %s = %p\n",
        nombre,
        puntero
    );

    return 1;
}


/* ============================================================================
 * TEST 01
 *
 * CREAR LAYER
 *
 * Comprueba la estructura creada por crearLayer().
 * ========================================================================== */

static void test_crearLayer(void)
{
    const char *nombre =
        "LAYER-01 - crearLayer crea correctamente la capa";

    float datosW[] = {
        0.10f, 0.20f,
        0.30f, 0.40f,
        0.50f, 0.60f
    };

    float datosB[] = {
        0.70f,
        0.80f,
        0.90f
    };

    struct matrix W;
    struct matrix B;

    struct layer *layer = NULL;

    int correcto = 1;
    int retorno;

    W.fil = 3;
    W.col = 2;
    W.datos = datosW;

    B.fil = 3;
    B.col = 1;
    B.datos = datosB;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    printf("Configuración de entrada:\n");
    imprimirMatriz("W", &W);
    imprimirMatriz("b", &B);
    printf("Activación solicitada: SIGMOID (%d)\n", SIGMOID);

    retorno = crearLayer(
        &W,
        &B,
        SIGMOID,
        &layer
    );

    printf("\nCódigo de retorno: %d\n", retorno);
    printf("Layer resultante:\n");

    imprimirLayer("layer", layer);

    if (layer == NULL)
    {
        printf(
            "[ERROR] crearLayer() devolvió NULL.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    if (retorno != 0)
    {
        printf(
            "[AVISO] crearLayer() devolvió %d.\n"
            "        Se continúa comprobando la estructura.\n",
            retorno
        );
    }

    if (layer->nEntradas != 2)
    {
        printf(
            "[ERROR] nEntradas incorrecto.\n"
            "        Esperado: 2\n"
            "        Obtenido : %d\n",
            layer->nEntradas
        );

        correcto = 0;
    }
    else
    {
        printf("[OK] nEntradas = 2.\n");
    }

    if (layer->nSalidas != 3)
    {
        printf(
            "[ERROR] nSalidas incorrecto.\n"
            "        Esperado: 3\n"
            "        Obtenido : %d\n",
            layer->nSalidas
        );

        correcto = 0;
    }
    else
    {
        printf("[OK] nSalidas = 3.\n");
    }

    if (layer->activacion != SIGMOID)
    {
        printf(
            "[ERROR] Activación incorrecta.\n"
            "        Esperada: SIGMOID (%d)\n"
            "        Obtenida : %d\n",
            SIGMOID,
            layer->activacion
        );

        correcto = 0;
    }
    else
    {
        printf("[OK] Activación = SIGMOID.\n");
    }

    if (layer->pesos == NULL)
    {
        printf(
            "[ERROR] layer->pesos == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        if (!comprobarMatriz(
                "layer->pesos",
                layer->pesos,
                datosW,
                3,
                2))
        {
            correcto = 0;
        }
    }

    if (layer->bias == NULL)
    {
        printf(
            "[ERROR] layer->bias == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        if (!comprobarMatriz(
                "layer->bias",
                layer->bias,
                datosB,
                3,
                1))
        {
            correcto = 0;
        }
    }

    if (layer->gradientes != NULL)
    {
        printf(
            "[ERROR] layer->gradientes no está NULL después de crear la capa.\n"
            "        Valor obtenido: %p\n",
            (void *)layer->gradientes
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] layer->gradientes está inicialmente NULL.\n"
        );
    }

    eliminarLayer(&layer);
    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 02
 *
 * CREAR LAYER CON TANH
 *
 * Comprueba que la activación no quede fijada a SIGMOID.
 * ========================================================================== */

static void test_crearLayer_tanh(void)
{
    const char *nombre =
        "LAYER-02 - crearLayer conserva activación TANH";

    float datosW[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float datosB[] = {
        0.0f,
        0.0f
    };

    struct matrix W = {2, 2, datosW};
    struct matrix B = {2, 1, datosB};

    struct layer *layer = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    printf("Activación solicitada: TANH (%d)\n", TANH);

    crearLayer(
        &W,
        &B,
        TANH,
        &layer
    );

    if (layer == NULL)
    {
        printf(
            "[ERROR] crearLayer() no creó la capa.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    printf(
        "Activación almacenada: %d\n",
        layer->activacion
    );

    if (layer->activacion != TANH)
    {
        printf(
            "[ERROR] Se esperaba TANH (%d), pero se obtuvo %d.\n",
            TANH,
            layer->activacion
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] La capa conserva TANH.\n"
        );
    }

    eliminarLayer(&layer);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 03
 *
 * CREAR LAYER CON RELU
 * ========================================================================== */

static void test_crearLayer_relu(void)
{
    const char *nombre =
        "LAYER-03 - crearLayer conserva activación RELU";

    float datosW[] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    float datosB[] = {
        1.0f,
        -1.0f
    };

    struct matrix W = {2, 2, datosW};
    struct matrix B = {2, 1, datosB};

    struct layer *layer = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    crearLayer(
        &W,
        &B,
        RELU,
        &layer
    );

    if (layer == NULL)
    {
        printf(
            "[ERROR] crearLayer() no creó la capa.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    if (layer->activacion != RELU)
    {
        printf(
            "[ERROR] Activación incorrecta.\n"
            "        Esperada: RELU (%d)\n"
            "        Obtenida : %d\n",
            RELU,
            layer->activacion
        );

        correcto = 0;
    }
    else
    {
        printf("[OK] Activación = RELU.\n");
    }

    eliminarLayer(&layer);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 04
 *
 * FORWARD SIGMOID
 *
 * Capa:
 *
 *       W = [  1   2 ]
 *           [ -1   1 ]
 *           [ 0.5 -1 ]
 *
 *       b = [ 0.5 ]
 *           [ 0.0 ]
 *           [-0.5 ]
 *
 *       entrada = [ 1 ]
 *                 [ 2 ]
 *
 *
 * Z = W*A+b
 *
 * z0 = 1*1 + 2*2 + 0.5 = 5.5
 * z1 = -1*1 + 1*2 + 0   = 1
 * z2 = 0.5*1 - 1*2 -0.5 = -2
 *
 * A = sigmoid(Z)
 * ========================================================================== */

static void test_forward_sigmoid(void)
{
    const char *nombre =
        "LAYER-04 - forward completo con SIGMOID";

    float datosW[] = {
         1.0f,  2.0f,
        -1.0f,  1.0f,
         0.5f, -1.0f
    };

    float datosB[] = {
         0.5f,
         0.0f,
        -0.5f
    };

    float datosEntrada[] = {
        1.0f,
        2.0f
    };

    float esperadoZ[] = {
         5.5f,
         1.0f,
        -2.0f
    };

    float esperadoA[] = {
        1.0f / (1.0f + expf(-5.5f)),
        1.0f / (1.0f + expf(-1.0f)),
        1.0f / (1.0f + expf( 2.0f))
    };

    struct matrix W = {3, 2, datosW};
    struct matrix B = {3, 1, datosB};
    struct matrix entrada = {2, 1, datosEntrada};

    struct matrix salida;

    struct layer *layer = NULL;
    struct cache *cache = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    imprimirMatriz("W", &W);
    imprimirMatriz("b", &B);
    imprimirMatriz("Entrada A", &entrada);

    printf("\nCálculo esperado:\n");
    printf("Z = W*A+b\n");
    printf("Z esperado = [5.5, 1.0, -2.0]^T\n");
    printf("A = sigmoid(Z)\n");

    crearLayer(
        &W,
        &B,
        SIGMOID,
        &layer
    );

    if (layer == NULL)
    {
        printf("[ERROR] No se pudo crear layer.\n");
        registrarResultado(nombre, 0);
        return;
    }

    /*
     * La estructura de salida se prepara con las dimensiones esperadas.
     */
    salida.fil = 3;
    salida.col = 1;

    salida.datos = malloc(
        3 * sizeof(float)
    );

    if (salida.datos == NULL)
    {
        printf("[ERROR] No se pudo reservar salida.\n");
        eliminarLayer(&layer);
        registrarResultado(nombre, 0);
        return;
    }

    printf("\nEjecutando forward()...\n");

    cache = forward(
        &entrada,
        layer,
        &salida
    );

    printf("\nCódigo/resultado de forward:\n");
    printf(
        "cache = %p\n",
        (void *)cache
    );

    printf("\nSalida obtenida:\n");
    imprimirMatriz("A salida", &salida);

    printf("\nSalida esperada:\n");

    {
        struct matrix temporal = {
            3,
            1,
            esperadoA
        };

        imprimirMatriz(
            "A esperada",
            &temporal
        );
    }

    if (!comprobarMatriz(
            "salida",
            &salida,
            esperadoA,
            3,
            1))
    {
        correcto = 0;
    }
    else
    {
        printf(
            "[OK] La activación de salida coincide con sigmoid(W*A+b).\n"
        );
    }

    if (cache == NULL)
    {
        printf(
            "\n[ERROR] forward() devolvió cache == NULL.\n"
            "        No se puede comprobar Z ni A almacenados.\n"
        );

        correcto = 0;
    }
    else
    {
        printf("\nCache generado por forward():\n");

        imprimirMatriz(
            "cache->z",
            cache->z
        );

        imprimirMatriz(
            "cache->a",
            cache->a
        );

        printf("\nComprobando Z almacenado...\n");

        if (!comprobarMatriz(
                "cache->z",
                cache->z,
                esperadoZ,
                3,
                1))
        {
            correcto = 0;
        }
        else
        {
            printf(
                "[OK] cache->z contiene W*A+b correctamente.\n"
            );
        }

        printf("\nComprobando A almacenada...\n");

        if (!comprobarMatriz(
                "cache->a",
                cache->a,
                esperadoA,
                3,
                1))
        {
            correcto = 0;
        }
        else
        {
            printf(
                "[OK] cache->a contiene sigmoid(Z) correctamente.\n"
            );
        }

        if (cache->a == &salida)
        {
            printf(
                "[INFO] cache->a apunta directamente a la estructura salida.\n"
            );
        }
    }

    if (cache != NULL)
        eliminarCacheLayer(&cache);

    free(salida.datos);

    eliminarLayer(&layer);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 05
 *
 * FORWARD TANH
 *
 * W = identidad
 * b = 0
 *
 * Por tanto:
 *
 *      Z = entrada
 *      A = tanh(entrada)
 * ========================================================================== */

static void test_forward_tanh(void)
{
    const char *nombre =
        "LAYER-05 - forward completo con TANH";

    float datosW[] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    float datosB[] = {
        0.0f,
        0.0f
    };

    float datosEntrada[] = {
        -1.0f,
         2.0f
    };

    float esperadoZ[] = {
        -1.0f,
         2.0f
    };

    float esperadoA[] = {
        tanhf(-1.0f),
        tanhf( 2.0f)
    };

    struct matrix W = {2, 2, datosW};
    struct matrix B = {2, 1, datosB};
    struct matrix entrada = {2, 1, datosEntrada};

    struct matrix salida;

    struct layer *layer = NULL;
    struct cache *cache = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    imprimirMatriz("W", &W);
    imprimirMatriz("b", &B);
    imprimirMatriz("Entrada", &entrada);

    crearLayer(
        &W,
        &B,
        TANH,
        &layer
    );

    if (layer == NULL)
    {
        printf("[ERROR] No se pudo crear layer.\n");
        registrarResultado(nombre, 0);
        return;
    }

    salida.fil = 2;
    salida.col = 1;
    salida.datos = malloc(2 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR] No se pudo reservar salida.\n");
        eliminarLayer(&layer);
        registrarResultado(nombre, 0);
        return;
    }

    cache = forward(
        &entrada,
        layer,
        &salida
    );

    printf("\nSalida:\n");
    imprimirMatriz("A", &salida);

    if (!comprobarMatriz(
            "salida",
            &salida,
            esperadoA,
            2,
            1))
    {
        correcto = 0;
    }
    else
    {
        printf(
            "[OK] tanh(Z) coincide con la salida obtenida.\n"
        );
    }

    if (cache == NULL)
    {
        printf(
            "[ERROR] forward() no devolvió cache.\n"
        );

        correcto = 0;
    }
    else
    {
        printf("\nCache:\n");
        imprimirMatriz("cache->z", cache->z);
        imprimirMatriz("cache->a", cache->a);

        if (!comprobarMatriz(
                "cache->z",
                cache->z,
                esperadoZ,
                2,
                1))
        {
            correcto = 0;
        }

        if (!comprobarMatriz(
                "cache->a",
                cache->a,
                esperadoA,
                2,
                1))
        {
            correcto = 0;
        }
    }

    if (cache != NULL)
        eliminarCacheLayer(&cache);

    free(salida.datos);

    eliminarLayer(&layer);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 06
 *
 * FORWARD RELU
 *
 * W = identidad
 * b = 0
 *
 * entrada = [-2, 0, 3]
 *
 * Z = [-2, 0, 3]
 * A = [ 0, 0, 3]
 *
 * El punto 0 se incluye deliberadamente para comprobar la convención de la
 * implementación de ReLU.
 * ========================================================================== */

static void test_forward_relu(void)
{
    const char *nombre =
        "LAYER-06 - forward completo con RELU";

    float datosW[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };

    float datosB[] = {
        0.0f,
        0.0f,
        0.0f
    };

    float datosEntrada[] = {
        -2.0f,
         0.0f,
         3.0f
    };

    float esperadoZ[] = {
        -2.0f,
         0.0f,
         3.0f
    };

    float esperadoA[] = {
        0.0f,
        0.0f,
        3.0f
    };

    struct matrix W = {3, 3, datosW};
    struct matrix B = {3, 1, datosB};
    struct matrix entrada = {3, 1, datosEntrada};

    struct matrix salida;

    struct layer *layer = NULL;
    struct cache *cache = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    imprimirMatriz("Entrada", &entrada);

    printf(
        "\nCaso utilizado:\n"
        "  z < 0 -> ReLU(z) = 0\n"
        "  z = 0 -> se comprueba la convención de relu()\n"
        "  z > 0 -> ReLU(z) = z\n"
    );

    crearLayer(
        &W,
        &B,
        RELU,
        &layer
    );

    if (layer == NULL)
    {
        printf("[ERROR] No se pudo crear layer.\n");
        registrarResultado(nombre, 0);
        return;
    }

    salida.fil = 3;
    salida.col = 1;
    salida.datos = malloc(3 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR] No se pudo reservar salida.\n");
        eliminarLayer(&layer);
        registrarResultado(nombre, 0);
        return;
    }

    cache = forward(
        &entrada,
        layer,
        &salida
    );

    printf("\nSalida obtenida:\n");
    imprimirMatriz("A", &salida);

    printf("\nSalida esperada:\n");
    {
        struct matrix temporal = {
            3,
            1,
            esperadoA
        };

        imprimirMatriz(
            "A esperada",
            &temporal
        );
    }

    if (!comprobarMatriz(
            "salida",
            &salida,
            esperadoA,
            3,
            1))
    {
        correcto = 0;
    }

    if (cache == NULL)
    {
        printf(
            "[ERROR] forward() devolvió cache == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        if (!comprobarMatriz(
                "cache->z",
                cache->z,
                esperadoZ,
                3,
                1))
        {
            correcto = 0;
        }

        if (!comprobarMatriz(
                "cache->a",
                cache->a,
                esperadoA,
                3,
                1))
        {
            correcto = 0;
        }
    }

    if (cache != NULL)
        eliminarCacheLayer(&cache);

    free(salida.datos);

    eliminarLayer(&layer);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 07
 *
 * CAMBIO DE ENTRADA
 *
 * Se utiliza la misma capa para dos entradas diferentes.
 *
 * Esto comprueba que forward() no se quede con los valores del primer forward.
 * ========================================================================== */

static void test_forward_dos_entradas(void)
{
    const char *nombre =
        "LAYER-07 - dos forward consecutivos con entradas diferentes";

    float datosW[] = {
        2.0f
    };

    float datosB[] = {
        1.0f
    };

    float datosEntrada1[] = {
        2.0f
    };

    float datosEntrada2[] = {
        -3.0f
    };

    float esperadoA1[] = {
        1.0f / (1.0f + expf(-5.0f))
    };

    float esperadoA2[] = {
        1.0f / (1.0f + expf(5.0f))
    };

    struct matrix W = {1, 1, datosW};
    struct matrix B = {1, 1, datosB};

    struct matrix entrada1 = {
        1,
        1,
        datosEntrada1
    };

    struct matrix entrada2 = {
        1,
        1,
        datosEntrada2
    };

    struct matrix salida1;
    struct matrix salida2;

    struct layer *layer = NULL;

    struct cache *cache1 = NULL;
    struct cache *cache2 = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    crearLayer(
        &W,
        &B,
        SIGMOID,
        &layer
    );

    if (layer == NULL)
    {
        printf("[ERROR] No se pudo crear layer.\n");
        registrarResultado(nombre, 0);
        return;
    }

    salida1.fil = 1;
    salida1.col = 1;
    salida1.datos = malloc(sizeof(float));

    salida2.fil = 1;
    salida2.col = 1;
    salida2.datos = malloc(sizeof(float));

    if (salida1.datos == NULL ||
        salida2.datos == NULL)
    {
        printf("[ERROR] No se pudo reservar alguna salida.\n");

        free(salida1.datos);
        free(salida2.datos);

        eliminarLayer(&layer);

        registrarResultado(nombre, 0);
        return;
    }

    printf("PRIMER FORWARD\n");
    imprimirMatriz("Entrada 1", &entrada1);

    cache1 = forward(
        &entrada1,
        layer,
        &salida1
    );

    imprimirMatriz(
        "Salida 1",
        &salida1
    );

    if (!comprobarMatriz(
            "salida1",
            &salida1,
            esperadoA1,
            1,
            1))
    {
        correcto = 0;
    }

    if (cache1 == NULL)
    {
        printf(
            "[ERROR] Primer forward no produjo cache.\n"
        );

        correcto = 0;
    }

    printf("\nSEGUNDO FORWARD\n");
    imprimirMatriz("Entrada 2", &entrada2);

    cache2 = forward(
        &entrada2,
        layer,
        &salida2
    );

    imprimirMatriz(
        "Salida 2",
        &salida2
    );

    if (!comprobarMatriz(
            "salida2",
            &salida2,
            esperadoA2,
            1,
            1))
    {
        correcto = 0;
    }

    if (cache2 == NULL)
    {
        printf(
            "[ERROR] Segundo forward no produjo cache.\n"
        );

        correcto = 0;
    }

    /*
     * El segundo forward no debería haber alterado la salida1 ya calculada.
     */
    printf(
        "\nComprobando que la salida del primer forward sigue siendo válida...\n"
    );

    if (!comprobarMatriz(
            "salida1 después del segundo forward",
            &salida1,
            esperadoA1,
            1,
            1))
    {
        correcto = 0;
    }
    else
    {
        printf(
            "[OK] La salida1 no ha sido modificada por el segundo forward.\n"
        );
    }

    if (cache1 != NULL)
        eliminarCacheLayer(&cache1);

    if (cache2 != NULL)
        eliminarCacheLayer(&cache2);

    free(salida1.datos);
    free(salida2.datos);

    eliminarLayer(&layer);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 08
 *
 * CONSISTENCIA MANUAL DE Z
 *
 * Este test separa explícitamente:
 *
 *      multiplicación W*A
 *      suma del bias
 *
 * y comprueba que el Z almacenado sea exactamente el esperado.
 * ========================================================================== */

static void test_forward_z_manual(void)
{
    const char *nombre =
        "LAYER-08 - Z coincide exactamente con W*A+b";

    float datosW[] = {
         2.0f, -1.0f,
         0.5f,  3.0f
    };

    float datosB[] = {
         4.0f,
        -2.0f
    };

    float datosEntrada[] = {
         5.0f,
        -2.0f
    };

    /*
     * z0 = 2*5 + (-1)*(-2) + 4 = 16
     * z1 = 0.5*5 + 3*(-2) - 2 = -6.5
     */
    float esperadoZ[] = {
        16.0f,
        -5.5f
    };

    float esperadoA[] = {
        16.0f,
         0.0f
    };

    struct matrix W = {2, 2, datosW};
    struct matrix B = {2, 1, datosB};
    struct matrix entrada = {2, 1, datosEntrada};

    struct matrix salida;

    struct layer *layer = NULL;
    struct cache *cache = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    imprimirMatriz("W", &W);
    imprimirMatriz("b", &B);
    imprimirMatriz("A entrada", &entrada);

    printf(
        "\nCálculo manual:\n"
        "  z0 = 2*5 + (-1)*(-2) + 4 = 16\n"
        "  z1 = 0.5*5 + 3*(-2) - 2 = -5.5\n"
    );

    crearLayer(
        &W,
        &B,
        RELU,
        &layer
    );

    if (layer == NULL)
    {
        printf("[ERROR] No se pudo crear layer.\n");
        registrarResultado(nombre, 0);
        return;
    }

    salida.fil = 2;
    salida.col = 1;
    salida.datos = malloc(2 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR] No se pudo reservar salida.\n");
        eliminarLayer(&layer);
        registrarResultado(nombre, 0);
        return;
    }

    cache = forward(
        &entrada,
        layer,
        &salida
    );

    if (cache == NULL)
    {
        printf(
            "[ERROR] No se pudo obtener cache para comprobar Z.\n"
        );

        correcto = 0;
    }
    else
    {
        printf("\nZ calculado por forward:\n");
        imprimirMatriz("cache->z", cache->z);

        if (!comprobarMatriz(
                "cache->z",
                cache->z,
                esperadoZ,
                2,
                1))
        {
            correcto = 0;
        }
        else
        {
            printf(
                "[OK] Z coincide exactamente con W*A+b.\n"
            );
        }

        printf("\nA después de RELU:\n");
        imprimirMatriz("cache->a", cache->a);

        if (!comprobarMatriz(
                "cache->a",
                cache->a,
                esperadoA,
                2,
                1))
        {
            correcto = 0;
        }
    }

    if (cache != NULL)
        eliminarCacheLayer(&cache);

    free(salida.datos);

    eliminarLayer(&layer);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 09
 *
 * DIMENSIONES DE FORWARD
 *
 * 3 entradas -> 4 salidas.
 *
 * Se comprueba que:
 *
 *      W = 4x3
 *      b = 4x1
 *      A = 3x1
 *      salida = 4x1
 * ========================================================================== */

static void test_forward_dimensiones(void)
{
    const char *nombre =
        "LAYER-09 - forward respeta las dimensiones de la capa";

    float datosW[] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 1
    };

    float datosB[] = {
        0,
        0,
        0,
        0
    };

    float datosEntrada[] = {
        1,
        2,
        3
    };

    float esperado[] = {
        1,
        2,
        3,
        6
    };

    struct matrix W = {4, 3, datosW};
    struct matrix B = {4, 1, datosB};
    struct matrix entrada = {3, 1, datosEntrada};

    struct matrix salida;

    struct layer *layer = NULL;
    struct cache *cache = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    crearLayer(
        &W,
        &B,
        RELU,
        &layer
    );

    if (layer == NULL)
    {
        printf("[ERROR] No se pudo crear layer.\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf(
        "Configuración de capa:\n"
        "  nEntradas = %d\n"
        "  nSalidas  = %d\n",
        layer->nEntradas,
        layer->nSalidas
    );

    salida.fil = 4;
    salida.col = 1;
    salida.datos = malloc(4 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR] No se pudo reservar salida.\n");
        eliminarLayer(&layer);
        registrarResultado(nombre, 0);
        return;
    }

    cache = forward(
        &entrada,
        layer,
        &salida
    );

    printf("\nSalida obtenida:\n");
    imprimirMatriz("salida", &salida);

    if (!comprobarMatriz(
            "salida",
            &salida,
            esperado,
            4,
            1))
    {
        correcto = 0;
    }

    if (cache == NULL)
    {
        printf(
            "[ERROR] cache == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        if (cache->z == NULL)
        {
            printf(
                "[ERROR] cache->z == NULL.\n"
            );

            correcto = 0;
        }
        else if (cache->z->fil != 4 ||
                 cache->z->col != 1)
        {
            printf(
                "[ERROR] Dimensiones de cache->z incorrectas.\n"
                "        Esperadas: 4x1\n"
                "        Obtenidas : %dx%d\n",
                cache->z->fil,
                cache->z->col
            );

            correcto = 0;
        }
        else
        {
            printf(
                "[OK] cache->z tiene dimensiones 4x1.\n"
            );
        }

        if (cache->a == NULL)
        {
            printf(
                "[ERROR] cache->a == NULL.\n"
            );

            correcto = 0;
        }
        else if (cache->a->fil != 4 ||
                 cache->a->col != 1)
        {
            printf(
                "[ERROR] Dimensiones de cache->a incorrectas.\n"
                "        Esperadas: 4x1\n"
                "        Obtenidas : %dx%d\n",
                cache->a->fil,
                cache->a->col
            );

            correcto = 0;
        }
        else
        {
            printf(
                "[OK] cache->a tiene dimensiones 4x1.\n"
            );
        }
    }

    if (cache != NULL)
        eliminarCacheLayer(&cache);

    free(salida.datos);

    eliminarLayer(&layer);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 10
 *
 * INDEPENDENCIA DE LA CAPA RESPECTO A W/B ORIGINALES
 *
 * Se modifican W y b después de crear la capa.
 *
 * Este test determina si crearLayer() hace copia de las matrices o comparte
 * los buffers originales.
 *
 * Para una estructura de red estable, normalmente es deseable que la capa sea
 * propietaria de sus pesos y bias.
 * ========================================================================== */

static void test_independencia_W_B(void)
{
    const char *nombre =
        "LAYER-10 - pesos y bias almacenados independientemente";

    float datosW[] = {
        1.0f
    };

    float datosB[] = {
        2.0f
    };

    float esperadoW[] = {
        1.0f
    };

    float esperadoB[] = {
        2.0f
    };

    struct matrix W = {1, 1, datosW};
    struct matrix B = {1, 1, datosB};

    struct layer *layer = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    crearLayer(
        &W,
        &B,
        RELU,
        &layer
    );

    if (layer == NULL)
    {
        printf("[ERROR] No se pudo crear layer.\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("Pesos antes de modificar originales:\n");
    imprimirMatriz("W", &W);
    imprimirMatriz("layer->pesos", layer->pesos);

    printf("\nBias antes de modificar originales:\n");
    imprimirMatriz("B", &B);
    imprimirMatriz("layer->bias", layer->bias);

    datosW[0] = 100.0f;
    datosB[0] = 200.0f;

    printf("\nDespués de modificar las matrices originales:\n");
    imprimirMatriz("W original", &W);
    imprimirMatriz("layer->pesos", layer->pesos);

    imprimirMatriz("B original", &B);
    imprimirMatriz("layer->bias", layer->bias);

    if (!comprobarMatriz(
            "layer->pesos",
            layer->pesos,
            esperadoW,
            1,
            1))
    {
        printf(
            "[ERROR] layer->pesos depende del buffer externo de W.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] layer->pesos conserva su valor original.\n"
        );
    }

    if (!comprobarMatriz(
            "layer->bias",
            layer->bias,
            esperadoB,
            1,
            1))
    {
        printf(
            "[ERROR] layer->bias depende del buffer externo de B.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] layer->bias conserva su valor original.\n"
        );
    }

    eliminarLayer(&layer);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 11
 *
 * ELIMINACIÓN DE LAYER
 *
 * Se comprueba que eliminarLayer(&layer) deje el puntero a NULL.
 * ========================================================================== */

static void test_eliminarLayer(void)
{
    const char *nombre =
        "LAYER-11 - eliminarLayer libera y anula la capa";

    float datosW[] = {
        1.0f, 2.0f
    };

    float datosB[] = {
        3.0f
    };

    struct matrix W = {1, 2, datosW};
    struct matrix B = {1, 1, datosB};

    struct layer *layer = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    crearLayer(
        &W,
        &B,
        SIGMOID,
        &layer
    );

    if (layer == NULL)
    {
        printf(
            "[ERROR] No se pudo crear la capa para el test.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    printf(
        "Layer antes de eliminar: %p\n",
        (void *)layer
    );

    printf(
        "Pesos: %p\n",
        (void *)layer->pesos
    );

    printf(
        "Bias : %p\n",
        (void *)layer->bias
    );

    printf("\nEjecutando eliminarLayer(&layer)...\n");

    eliminarLayer(&layer);

    printf(
        "Layer después de eliminar: %p\n",
        (void *)layer
    );

    if (layer != NULL)
    {
        printf(
            "[ERROR] eliminarLayer() no dejó layer == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] layer == NULL después de eliminarLayer().\n"
        );
    }

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 12
 *
 * CREAR Y ELIMINAR VARIAS CAPAS
 *
 * Comprueba que no haya problemas evidentes al crear sucesivamente capas
 * con diferentes configuraciones.
 * ========================================================================== */

static void test_crear_eliminar_varias(void)
{
    const char *nombre =
        "LAYER-12 - creación y eliminación repetida de capas";

    float W1datos[] = {1.0f};
    float B1datos[] = {0.0f};

    float W2datos[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float B2datos[] = {
        0.0f,
        1.0f
    };

    float W3datos[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };

    float B3datos[] = {
        1.0f,
        2.0f,
        3.0f
    };

    struct matrix W1 = {1, 1, W1datos};
    struct matrix B1 = {1, 1, B1datos};

    struct matrix W2 = {2, 2, W2datos};
    struct matrix B2 = {2, 1, B2datos};

    struct matrix W3 = {3, 2, W3datos};
    struct matrix B3 = {3, 1, B3datos};

    struct layer *l1 = NULL;
    struct layer *l2 = NULL;
    struct layer *l3 = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    printf("Creando capa 1: 1 -> 1, SIGMOID\n");

    crearLayer(
        &W1,
        &B1,
        SIGMOID,
        &l1
    );

    printf("Creando capa 2: 2 -> 2, TANH\n");

    crearLayer(
        &W2,
        &B2,
        TANH,
        &l2
    );

    printf("Creando capa 3: 2 -> 3, RELU\n");

    crearLayer(
        &W3,
        &B3,
        RELU,
        &l3
    );

    if (l1 == NULL)
    {
        printf("[ERROR] No se creó l1.\n");
        correcto = 0;
    }
    else
    {
        printf("[OK] l1 creada.\n");
    }

    if (l2 == NULL)
    {
        printf("[ERROR] No se creó l2.\n");
        correcto = 0;
    }
    else
    {
        printf("[OK] l2 creada.\n");
    }

    if (l3 == NULL)
    {
        printf("[ERROR] No se creó l3.\n");
        correcto = 0;
    }
    else
    {
        printf("[OK] l3 creada.\n");
    }

    printf("\nEliminando las tres capas...\n");

    if (l1 != NULL)
        eliminarLayer(&l1);

    if (l2 != NULL)
        eliminarLayer(&l2);

    if (l3 != NULL)
        eliminarLayer(&l3);

    printf(
        "l1 = %p\n"
        "l2 = %p\n"
        "l3 = %p\n",
        (void *)l1,
        (void *)l2,
        (void *)l3
    );

    if (l1 != NULL ||
        l2 != NULL ||
        l3 != NULL)
    {
        printf(
            "[ERROR] Alguna capa no quedó NULL después de eliminarla.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] Las tres capas fueron eliminadas correctamente.\n"
        );
    }

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * MAIN
 * ========================================================================== */

int main(void)
{
    printf("\n");
    printf("===============================================================================\n");
    printf("                                TEST LAYER\n");
    printf("===============================================================================\n");
    printf("\n");

    printf("Funciones evaluadas:\n");
    printf("  - crearLayer()\n");
    printf("  - eliminarLayer()\n");
    printf("  - forward()\n");
    printf("  - inicializarCache() [indirectamente mediante forward]\n");
    printf("  - guardarCacheLayerA() [indirectamente mediante forward]\n");
    printf("  - guardarCacheLayerZ() [indirectamente mediante forward]\n");
    printf("\n");

    printf(
        "Modelo esperado de una capa:\n"
        "  Z = W*A+b\n"
        "  A = f(Z)\n"
    );

    printf("\n");
    printf(
        "Tolerancia absoluta: %.2e\n"
        "Tolerancia relativa: %.2e\n",
        TOL_ABS,
        TOL_REL
    );

    separador();

    test_crearLayer();

    separador();

    test_crearLayer_tanh();
    test_crearLayer_relu();

    separador();

    test_forward_sigmoid();
    test_forward_tanh();
    test_forward_relu();

    separador();

    test_forward_dos_entradas();

    separador();

    test_forward_z_manual();

    separador();

    test_forward_dimensiones();

    separador();

    test_independencia_W_B();

    separador();

    test_eliminarLayer();

    separador();

    test_crear_eliminar_varias();

    separador();

    printf("\n");
    printf("===============================================================================\n");
    printf("                                  RESUMEN\n");
    printf("===============================================================================\n");

    printf(
        "Tests ejecutados : %d\n"
        "Tests pasados    : %d\n"
        "Tests fallados   : %d\n",
        tests_totales,
        tests_pasados,
        tests_fallados
    );

    printf("\n");

    if (tests_fallados == 0)
    {
        printf(
            "RESULTADO GLOBAL: PASS\n"
            "Todos los tests de layer han pasado correctamente.\n"
        );
    }
    else
    {
        printf(
            "RESULTADO GLOBAL: FAIL\n"
            "Se han encontrado %d test(s) con errores.\n"
            "Revisa el detalle mostrado en cada test.\n",
            tests_fallados
        );
    }

    printf("===============================================================================\n");

    return tests_fallados == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}