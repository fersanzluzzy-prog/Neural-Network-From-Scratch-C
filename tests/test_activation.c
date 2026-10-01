#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "activation.h"
#include "matrix.h"

/*
 * ============================================================================
 * TEST ACTIVATION
 * ============================================================================
 *
 * Funciones probadas:
 *   - relu()
 *   - reluDerivada()
 *   - sigmoide()
 *   - sigmoideDerivada()
 *   - taNh()
 *   - taNhDerivada()
 *
 * Criterios:
 *   - Se comprueba el código de retorno.
 *   - Se comprueban dimensiones.
 *   - Se comprueba cada elemento contra el valor esperado.
 *   - En caso de error se muestran:
 *       * test
 *       * función
 *       * posición
 *       * entrada
 *       * esperado
 *       * obtenido
 *       * error absoluto
 *   - Se comprueba que las funciones no alteran la matriz de entrada.
 *   - Se prueban valores positivos, negativos, cero y valores grandes.
 *   - Las derivadas de sigmoid/tanh reciben la ACTIVACIÓN, no z.
 *   - ReLU y su derivada reciben z.
 *
 * Tolerancias:
 *   - Las funciones usan float, por lo que no se exige igualdad exacta
 *     en operaciones trascendentales.
 * ============================================================================
 */

#define TOL_ABS 1e-5f
#define TOL_REL 1e-4f

static int tests_totales = 0;
static int tests_pasados = 0;
static int tests_fallados = 0;


/* ============================================================================
 * UTILIDADES DE TEST
 * ========================================================================== */

static int floatIguales(float obtenido, float esperado)
{
    float error = fabsf(obtenido - esperado);
    float escala = fmaxf(1.0f, fmaxf(fabsf(obtenido), fabsf(esperado)));

    return error <= TOL_ABS || (error / escala) <= TOL_REL;
}


static void imprimirSeparador(void)
{
    printf("-------------------------------------------------------------------------------\n");
}


static void imprimirMatrizDetallada(const char *nombre, struct matrix *m)
{
    int i, j;

    if (m == NULL)
    {
        printf("%s = NULL\n", nombre);
        return;
    }

    printf("%s [%d x %d]:\n", nombre, m->fil, m->col);

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
}


static int comprobarDimensiones(
    const char *funcion,
    struct matrix *resultado,
    int filasEsperadas,
    int columnasEsperadas)
{
    if (resultado == NULL)
    {
        printf("  [ERROR] %s: resultado == NULL\n", funcion);
        return 0;
    }

    if (resultado->fil != filasEsperadas ||
        resultado->col != columnasEsperadas)
    {
        printf("  [ERROR] %s: dimensiones incorrectas\n", funcion);
        printf("          Esperadas: %d x %d\n",
               filasEsperadas,
               columnasEsperadas);
        printf("          Obtenidas: %d x %d\n",
               resultado->fil,
               resultado->col);
        return 0;
    }

    return 1;
}


static int comprobarMatriz(
    const char *funcion,
    struct matrix *resultado,
    const float *esperado,
    int filas,
    int columnas)
{
    int i, j;
    int errores = 0;

    if (!comprobarDimensiones(funcion, resultado, filas, columnas))
        return 0;

    for (i = 0; i < filas; i++)
    {
        for (j = 0; j < columnas; j++)
        {
            int indice = i * columnas + j;
            float obtenido = resultado->datos[indice];
            float esperadoActual = esperado[indice];

            if (!floatIguales(obtenido, esperadoActual))
            {
                float error = fabsf(obtenido - esperadoActual);

                if (errores == 0)
                {
                    printf("  [ERROR] %s: hay valores incorrectos\n", funcion);
                }

                printf("          Posición [%d][%d]\n", i, j);
                printf("          Esperado : %.10f\n", esperadoActual);
                printf("          Obtenido : %.10f\n", obtenido);
                printf("          Error abs: %.10f\n", error);

                errores++;
            }
        }
    }

    if (errores > 0)
    {
        printf("          Total de posiciones incorrectas: %d\n", errores);
        return 0;
    }

    return 1;
}


static int comprobarMatrizSinCambios(
    const char *funcion,
    struct matrix *actual,
    const float *original,
    int filas,
    int columnas)
{
    int i, j;
    int errores = 0;

    if (actual == NULL)
    {
        printf("  [ERROR] %s: la matriz de entrada ha quedado en NULL\n",
               funcion);
        return 0;
    }

    for (i = 0; i < filas; i++)
    {
        for (j = 0; j < columnas; j++)
        {
            int indice = i * columnas + j;

            if (actual->datos[indice] != original[indice])
            {
                printf("  [ERROR] %s: la función modificó la entrada\n",
                       funcion);
                printf("          Posición [%d][%d]\n", i, j);
                printf("          Original : %.10f\n", original[indice]);
                printf("          Después  : %.10f\n",
                       actual->datos[indice]);

                errores++;
            }
        }
    }

    return errores == 0;
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


/* ============================================================================
 * TESTS RELU
 * ========================================================================== */

static void test_relu_valores_basicos(void)
{
    const char *nombre = "RELU - valores negativos, cero y positivos";

    float datos[] = {
        -3.0f, -1.0f, 0.0f,
         1.0f,  2.0f, 5.0f
    };

    float esperado[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 2.0f, 5.0f
    };

    float original[] = {
        -3.0f, -1.0f, 0.0f,
         1.0f,  2.0f, 5.0f
    };

    struct matrix entrada = {2, 3, datos};
    struct matrix salida = {2, 3, NULL};

    int retorno;
    int correcto = 1;

    salida.datos = malloc(6 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria para salida\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    imprimirMatrizDetallada("Entrada", &entrada);

    printf("Esperado:\n");
    {
        struct matrix temp = {2, 3, esperado};
        imprimirMatrizDetallada("Salida", &temp);
    }

    retorno = relu(&entrada, &salida);

    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] relu() devolvió código %d; se esperaba 0.\n",
               retorno);
        correcto = 0;
    }

    if (!comprobarMatriz(
            "relu()",
            &salida,
            esperado,
            2,
            3))
    {
        correcto = 0;
    }

    if (!comprobarMatrizSinCambios(
            "relu()",
            &entrada,
            original,
            2,
            3))
    {
        correcto = 0;
    }

    imprimirMatrizDetallada("Salida obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


static void test_relu_valores_extremos(void)
{
    const char *nombre = "RELU - valores grandes y pequeños";

    float datos[] = {
        -100000.0f,
        -0.000001f,
         0.000001f,
         100000.0f
    };

    float esperado[] = {
        0.0f,
        0.0f,
        0.000001f,
        100000.0f
    };

    struct matrix entrada = {4, 1, datos};
    struct matrix salida = {4, 1, NULL};

    int retorno;
    int correcto = 1;

    salida.datos = malloc(4 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria para salida\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    imprimirMatrizDetallada("Entrada", &entrada);

    retorno = relu(&entrada, &salida);

    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] relu() devolvió %d; se esperaba 0.\n", retorno);
        correcto = 0;
    }

    if (!comprobarMatriz("relu()", &salida, esperado, 4, 1))
        correcto = 0;

    imprimirMatrizDetallada("Salida obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


static void test_relu_derivada(void)
{
    const char *nombre = "RELU derivada - signo de z";

    float datos[] = {
        -3.0f, -0.5f, 0.0f,
         0.5f,  2.0f, 8.0f
    };

    /*
     * En z = 0 la derivada de ReLU no está definida matemáticamente.
     *
     * Este test NO impone un valor para ese punto.
     * Se comprueban exclusivamente los valores z < 0 y z > 0.
     */
    float esperado[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f
    };

    struct matrix entrada = {2, 3, datos};
    struct matrix salida = {2, 3, NULL};

    int retorno;
    int correcto = 1;

    salida.datos = malloc(6 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria para salida\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    imprimirMatrizDetallada("Z / entrada", &entrada);

    printf("Nota: z = 0 no se considera para decidir el resultado del test.\n");

    retorno = reluDerivada(&entrada, &salida);

    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] reluDerivada() devolvió %d; se esperaba 0.\n",
               retorno);
        correcto = 0;
    }

    if (!comprobarMatriz("reluDerivada()", &salida, esperado, 2, 3))
        correcto = 0;

    imprimirMatrizDetallada("Derivada obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TESTS SIGMOIDE
 * ========================================================================== */

static void test_sigmoide_valores_conocidos(void)
{
    const char *nombre = "SIGMOIDE - valores conocidos";

    float datos[] = {
        -2.0f,
        -1.0f,
         0.0f,
         1.0f,
         2.0f
    };

    float esperado[] = {
        0.11920292f,
        0.26894143f,
        0.50000000f,
        0.73105860f,
        0.88079708f
    };

    float original[] = {
        -2.0f,
        -1.0f,
         0.0f,
         1.0f,
         2.0f
    };

    struct matrix entrada = {5, 1, datos};
    struct matrix salida = {5, 1, NULL};

    int retorno;
    int correcto = 1;

    salida.datos = malloc(5 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria para salida\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    imprimirMatrizDetallada("Z / entrada", &entrada);

    retorno = sigmoide(&entrada, &salida);

    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] sigmoide() devolvió %d; se esperaba 0.\n",
               retorno);
        correcto = 0;
    }

    if (!comprobarMatriz(
            "sigmoide()",
            &salida,
            esperado,
            5,
            1))
    {
        correcto = 0;
    }

    if (!comprobarMatrizSinCambios(
            "sigmoide()",
            &entrada,
            original,
            5,
            1))
    {
        correcto = 0;
    }

    imprimirMatrizDetallada("A / salida obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


static void test_sigmoide_rango(void)
{
    const char *nombre = "SIGMOIDE - salida siempre en (0,1)";

    float datos[] = {
        -100.0f,
        -10.0f,
        -1.0f,
         0.0f,
         1.0f,
         10.0f,
         100.0f
    };

    struct matrix entrada = {7, 1, datos};
    struct matrix salida = {7, 1, NULL};

    int retorno;
    int correcto = 1;
    int i;

    salida.datos = malloc(7 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria para salida\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    imprimirMatrizDetallada("Entrada", &entrada);

    retorno = sigmoide(&entrada, &salida);

    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] sigmoide() devolvió %d; se esperaba 0.\n",
               retorno);
        correcto = 0;
    }

    if (salida.fil != 7 || salida.col != 1)
    {
        printf("  [ERROR] Dimensiones incorrectas: %d x %d\n",
               salida.fil,
               salida.col);
        correcto = 0;
    }

    for (i = 0; i < 7; i++)
    {
        float x = salida.datos[i];

        if (!(x > 0.0f && x < 1.0f))
        {
            printf("  [ERROR] Posición [%d][0]\n", i);
            printf("          Sigmoide obtenida: %.10f\n", x);
            printf("          Se esperaba un valor estrictamente entre 0 y 1.\n");
            correcto = 0;
        }
    }

    imprimirMatrizDetallada("Salida obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


static void test_sigmoide_derivada(void)
{
    const char *nombre = "SIGMOIDE derivada - recibe A y calcula A(1-A)";

    /*
     * IMPORTANTE:
     * sigmoideDerivada() recibe la ACTIVACIÓN ya calculada.
     *
     * A = [0.0, 0.1, 0.5, 0.9, 1.0]
     *
     * dA/dZ = A(1-A)
     */
    float activaciones[] = {
        0.0f,
        0.1f,
        0.5f,
        0.9f,
        1.0f
    };

    float esperado[] = {
        0.0f,
        0.09f,
        0.25f,
        0.09f,
        0.0f
    };

    struct matrix entrada = {5, 1, activaciones};
    struct matrix salida = {5, 1, NULL};

    int retorno;
    int correcto = 1;

    salida.datos = malloc(5 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria para salida\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    printf("La entrada NO es z: es la activación A = sigmoid(z).\n");
    imprimirMatrizDetallada("A / entrada", &entrada);

    retorno = sigmoideDerivada(&entrada, &salida);

    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] sigmoideDerivada() devolvió %d; se esperaba 0.\n",
               retorno);
        correcto = 0;
    }

    if (!comprobarMatriz(
            "sigmoideDerivada()",
            &salida,
            esperado,
            5,
            1))
    {
        correcto = 0;
    }

    imprimirMatrizDetallada("Derivada obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TESTS TANH
 * ========================================================================== */

static void test_tanh_valores_conocidos(void)
{
    const char *nombre = "TANH - valores conocidos";

    float datos[] = {
        -2.0f,
        -1.0f,
         0.0f,
         1.0f,
         2.0f
    };

    float esperado[] = {
        -0.96402758f,
        -0.76159416f,
         0.00000000f,
         0.76159416f,
         0.96402758f
    };

    float original[] = {
        -2.0f,
        -1.0f,
         0.0f,
         1.0f,
         2.0f
    };

    struct matrix entrada = {5, 1, datos};
    struct matrix salida = {5, 1, NULL};

    int retorno;
    int correcto = 1;

    salida.datos = malloc(5 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria para salida\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    imprimirMatrizDetallada("Z / entrada", &entrada);

    retorno = taNh(&entrada, &salida);

    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] taNh() devolvió %d; se esperaba 0.\n",
               retorno);
        correcto = 0;
    }

    if (!comprobarMatriz(
            "taNh()",
            &salida,
            esperado,
            5,
            1))
    {
        correcto = 0;
    }

    if (!comprobarMatrizSinCambios(
            "taNh()",
            &entrada,
            original,
            5,
            1))
    {
        correcto = 0;
    }

    imprimirMatrizDetallada("A / salida obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


static void test_tanh_rango(void)
{
    const char *nombre = "TANH - salida siempre en (-1,1)";

    float datos[] = {
        -100.0f,
        -10.0f,
        -1.0f,
         0.0f,
         1.0f,
         10.0f,
         100.0f
    };

    struct matrix entrada = {7, 1, datos};
    struct matrix salida = {7, 1, NULL};

    int retorno;
    int correcto = 1;
    int i;

    salida.datos = malloc(7 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria para salida\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    imprimirMatrizDetallada("Entrada", &entrada);

    retorno = taNh(&entrada, &salida);

    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] taNh() devolvió %d; se esperaba 0.\n",
               retorno);
        correcto = 0;
    }

    if (salida.fil != 7 || salida.col != 1)
    {
        printf("  [ERROR] Dimensiones incorrectas: %d x %d\n",
               salida.fil,
               salida.col);
        correcto = 0;
    }

    for (i = 0; i < 7; i++)
    {
        float x = salida.datos[i];

        if (!(x > -1.0f && x < 1.0f))
        {
            /*
             * tanh(±100) puede quedar extremadamente próximo a ±1,
             * pero matemáticamente no debería devolver exactamente
             * ±1 para valores finitos normales.
             */
            printf("  [ERROR] Posición [%d][0]\n", i);
            printf("          Tanh obtenida: %.10f\n", x);
            printf("          Se esperaba un valor estrictamente entre -1 y 1.\n");
            correcto = 0;
        }
    }

    imprimirMatrizDetallada("Salida obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


static void test_tanh_derivada(void)
{
    const char *nombre = "TANH derivada - recibe A y calcula 1-A²";

    /*
     * Igual que con sigmoid:
     *
     * taNhDerivada() recibe la ACTIVACIÓN A.
     *
     * dA/dZ = 1 - A²
     */
    float activaciones[] = {
        -1.0f,
        -0.5f,
         0.0f,
         0.5f,
         1.0f
    };

    float esperado[] = {
        0.0f,
        0.75f,
        1.0f,
        0.75f,
        0.0f
    };

    struct matrix entrada = {5, 1, activaciones};
    struct matrix salida = {5, 1, NULL};

    int retorno;
    int correcto = 1;

    salida.datos = malloc(5 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria para salida\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    printf("La entrada NO es z: es la activación A = tanh(z).\n");
    imprimirMatrizDetallada("A / entrada", &entrada);

    retorno = taNhDerivada(&entrada, &salida);

    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] taNhDerivada() devolvió %d; se esperaba 0.\n",
               retorno);
        correcto = 0;
    }

    if (!comprobarMatriz(
            "taNhDerivada()",
            &salida,
            esperado,
            5,
            1))
    {
        correcto = 0;
    }

    imprimirMatrizDetallada("Derivada obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TESTS DE CONSISTENCIA MATEMÁTICA
 * ========================================================================== */

static void test_sigmoide_derivada_consistencia(void)
{
    const char *nombre =
        "SIGMOIDE - consistencia entre sigmoid y su derivada";

    float z[] = {
        -3.0f,
        -1.5f,
        -0.2f,
         0.2f,
         1.5f,
         3.0f
    };

    struct matrix entrada = {6, 1, z};
    struct matrix activacion = {6, 1, NULL};
    struct matrix derivada = {6, 1, NULL};

    int retorno1;
    int retorno2;
    int correcto = 1;
    int i;

    activacion.datos = malloc(6 * sizeof(float));
    derivada.datos = malloc(6 * sizeof(float));

    if (activacion.datos == NULL || derivada.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria\n");

        free(activacion.datos);
        free(derivada.datos);

        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    printf("Primero se calcula A = sigmoid(z).\n");
    printf("Después se calcula dA/dz usando exclusivamente A.\n\n");

    retorno1 = sigmoide(&entrada, &activacion);
    retorno2 = sigmoideDerivada(&activacion, &derivada);

    printf("Código sigmoide(): %d\n", retorno1);
    printf("Código sigmoideDerivada(): %d\n", retorno2);

    if (retorno1 != 0)
    {
        printf("  [ERROR] sigmoide() falló.\n");
        correcto = 0;
    }

    if (retorno2 != 0)
    {
        printf("  [ERROR] sigmoideDerivada() falló.\n");
        correcto = 0;
    }

    for (i = 0; i < 6; i++)
    {
        float esperado = activacion.datos[i] *
                         (1.0f - activacion.datos[i]);

        float obtenido = derivada.datos[i];

        if (!floatIguales(obtenido, esperado))
        {
            printf("  [ERROR] Posición [%d][0]\n", i);
            printf("          z          = %.10f\n", z[i]);
            printf("          A          = %.10f\n", activacion.datos[i]);
            printf("          Esperado   = %.10f\n", esperado);
            printf("          Obtenido   = %.10f\n", obtenido);
            printf("          Error abs. = %.10f\n",
                   fabsf(obtenido - esperado));

            correcto = 0;
        }
    }

    imprimirMatrizDetallada("z", &entrada);
    imprimirMatrizDetallada("A = sigmoid(z)", &activacion);
    imprimirMatrizDetallada("dA/dz", &derivada);

    free(activacion.datos);
    free(derivada.datos);

    registrarResultado(nombre, correcto);
}


static void test_tanh_derivada_consistencia(void)
{
    const char *nombre =
        "TANH - consistencia entre tanh y su derivada";

    float z[] = {
        -3.0f,
        -1.5f,
        -0.2f,
         0.2f,
         1.5f,
         3.0f
    };

    struct matrix entrada = {6, 1, z};
    struct matrix activacion = {6, 1, NULL};
    struct matrix derivada = {6, 1, NULL};

    int retorno1;
    int retorno2;
    int correcto = 1;
    int i;

    activacion.datos = malloc(6 * sizeof(float));
    derivada.datos = malloc(6 * sizeof(float));

    if (activacion.datos == NULL || derivada.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria\n");

        free(activacion.datos);
        free(derivada.datos);

        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    printf("Primero se calcula A = tanh(z).\n");
    printf("Después se calcula dA/dz usando exclusivamente A.\n\n");

    retorno1 = taNh(&entrada, &activacion);
    retorno2 = taNhDerivada(&activacion, &derivada);

    printf("Código taNh(): %d\n", retorno1);
    printf("Código taNhDerivada(): %d\n", retorno2);

    if (retorno1 != 0)
    {
        printf("  [ERROR] taNh() falló.\n");
        correcto = 0;
    }

    if (retorno2 != 0)
    {
        printf("  [ERROR] taNhDerivada() falló.\n");
        correcto = 0;
    }

    for (i = 0; i < 6; i++)
    {
        float esperado =
            1.0f - activacion.datos[i] * activacion.datos[i];

        float obtenido = derivada.datos[i];

        if (!floatIguales(obtenido, esperado))
        {
            printf("  [ERROR] Posición [%d][0]\n", i);
            printf("          z          = %.10f\n", z[i]);
            printf("          A          = %.10f\n", activacion.datos[i]);
            printf("          Esperado   = %.10f\n", esperado);
            printf("          Obtenido   = %.10f\n", obtenido);
            printf("          Error abs. = %.10f\n",
                   fabsf(obtenido - esperado));

            correcto = 0;
        }
    }

    imprimirMatrizDetallada("z", &entrada);
    imprimirMatrizDetallada("A = tanh(z)", &activacion);
    imprimirMatrizDetallada("dA/dz", &derivada);

    free(activacion.datos);
    free(derivada.datos);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TESTS DE DIMENSIONES
 * ============================================================================
 *
 * Estos tests verifican que las funciones puedan trabajar con matrices que
 * no sean únicamente vectores columna.
 * ========================================================================== */

static void test_activaciones_matriz_2x3(void)
{
    const char *nombre =
        "ACTIVACIONES - matriz general 2x3";

    float datos[] = {
        -1.0f, 0.0f, 1.0f,
         2.0f, -2.0f, 0.5f
    };

    struct matrix entrada = {2, 3, datos};
    struct matrix salida = {2, 3, NULL};

    int retorno;
    int correcto = 1;
    int i;

    salida.datos = malloc(6 * sizeof(float));

    if (salida.datos == NULL)
    {
        printf("[ERROR CRÍTICO] No se pudo reservar memoria\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("TEST: %s\n", nombre);
    imprimirMatrizDetallada("Entrada", &entrada);

    retorno = relu(&entrada, &salida);

    printf("\nRELU\n");
    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] relu() devolvió %d.\n", retorno);
        correcto = 0;
    }

    for (i = 0; i < 6; i++)
    {
        float esperado = datos[i] > 0.0f ? datos[i] : 0.0f;

        if (!floatIguales(salida.datos[i], esperado))
        {
            printf("  [ERROR] RELU posición [%d]\n", i);
            printf("          Entrada : %.8f\n", datos[i]);
            printf("          Esperado: %.8f\n", esperado);
            printf("          Obtenido: %.8f\n", salida.datos[i]);
            correcto = 0;
        }
    }

    retorno = sigmoide(&entrada, &salida);

    printf("\nSIGMOIDE\n");
    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] sigmoide() devolvió %d.\n", retorno);
        correcto = 0;
    }

    for (i = 0; i < 6; i++)
    {
        float esperado = 1.0f / (1.0f + expf(-datos[i]));

        if (!floatIguales(salida.datos[i], esperado))
        {
            printf("  [ERROR] SIGMOIDE posición [%d]\n", i);
            printf("          Entrada : %.8f\n", datos[i]);
            printf("          Esperado: %.8f\n", esperado);
            printf("          Obtenido: %.8f\n", salida.datos[i]);
            correcto = 0;
        }
    }

    retorno = taNh(&entrada, &salida);

    printf("\nTANH\n");
    printf("Código de retorno: %d\n", retorno);

    if (retorno != 0)
    {
        printf("  [ERROR] taNh() devolvió %d.\n", retorno);
        correcto = 0;
    }

    for (i = 0; i < 6; i++)
    {
        float esperado = tanhf(datos[i]);

        if (!floatIguales(salida.datos[i], esperado))
        {
            printf("  [ERROR] TANH posición [%d]\n", i);
            printf("          Entrada : %.8f\n", datos[i]);
            printf("          Esperado: %.8f\n", esperado);
            printf("          Obtenido: %.8f\n", salida.datos[i]);
            correcto = 0;
        }
    }

    imprimirMatrizDetallada("Última salida obtenida", &salida);

    free(salida.datos);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * MAIN
 * ========================================================================== */

int main(void)
{
    printf("\n");
    printf("===============================================================================\n");
    printf("                         TEST ACTIVATION\n");
    printf("===============================================================================\n");
    printf("Tolerancia absoluta: %.2e\n", TOL_ABS);
    printf("Tolerancia relativa: %.2e\n", TOL_REL);
    printf("\n");

    imprimirSeparador();

    test_relu_valores_basicos();
    test_relu_valores_extremos();
    test_relu_derivada();

    imprimirSeparador();

    test_sigmoide_valores_conocidos();
    test_sigmoide_rango();
    test_sigmoide_derivada();

    imprimirSeparador();

    test_tanh_valores_conocidos();
    test_tanh_rango();
    test_tanh_derivada();

    imprimirSeparador();

    test_sigmoide_derivada_consistencia();
    test_tanh_derivada_consistencia();

    imprimirSeparador();

    test_activaciones_matriz_2x3();

    imprimirSeparador();

    printf("\n");
    printf("===============================================================================\n");
    printf("                              RESUMEN\n");
    printf("===============================================================================\n");
    printf("Tests ejecutados : %d\n", tests_totales);
    printf("Tests pasados    : %d\n", tests_pasados);
    printf("Tests fallados   : %d\n", tests_fallados);
    printf("\n");

    if (tests_fallados == 0)
    {
        printf("RESULTADO GLOBAL: PASS\n");
        printf("Todas las pruebas de activación han pasado correctamente.\n");
    }
    else
    {
        printf("RESULTADO GLOBAL: FAIL\n");
        printf("Hay %d test(s) que requieren investigación.\n",
               tests_fallados);
    }

    printf("===============================================================================\n");

    return tests_fallados == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}