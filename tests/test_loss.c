#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"
#include "loss.h"

#define TOLERANCIA 1e-5f
#define TOLERANCIA_NUMERICA 1e-4f

static int tests_pasados = 0;
static int tests_fallados = 0;


/* ============================================================
 * UTILIDADES
 * ============================================================ */

static int casiIgual(float a, float b, float tolerancia)
{
    float diferencia = fabsf(a - b);
    float escala = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));

    return diferencia <= tolerancia * escala;
}


static void imprimirMatrizTest(const char *nombre, const struct matrix *m)
{
    if (m == NULL)
    {
        printf("  %s: NULL\n", nombre);
        return;
    }

    printf("  %s [%dx%d]:\n",
           nombre,
           m->fil,
           m->col);

    if (m->datos == NULL)
    {
        printf("    datos = NULL\n");
        return;
    }

    for (int i = 0; i < m->fil; i++)
    {
        printf("    [ ");

        for (int j = 0; j < m->col; j++)
        {
            printf("%.9g", m->datos[i * m->col + j]);

            if (j < m->col - 1)
                printf(", ");
        }

        printf(" ]\n");
    }
}


static int comprobarMatriz(
    const char *test,
    const char *descripcion,
    struct matrix *actual,
    int filasEsperadas,
    int columnasEsperadas,
    const float *esperado)
{
    int fallo = 0;

    if (actual == NULL)
    {
        printf("\n[FAIL] %s\n", test);
        printf("  %s\n", descripcion);
        printf("  Resultado: puntero NULL\n");

        tests_fallados++;
        return 0;
    }

    if (actual->fil != filasEsperadas ||
        actual->col != columnasEsperadas)
    {
        printf("\n[FAIL] %s\n", test);
        printf("  %s\n", descripcion);

        printf("  Dimensiones esperadas: %dx%d\n",
               filasEsperadas,
               columnasEsperadas);

        printf("  Dimensiones obtenidas: %dx%d\n",
               actual->fil,
               actual->col);

        fallo = 1;
    }

    if (actual->datos == NULL)
    {
        printf("  Datos: NULL\n");
        fallo = 1;
    }

    if (!fallo)
    {
        for (int i = 0; i < filasEsperadas; i++)
        {
            for (int j = 0; j < columnasEsperadas; j++)
            {
                int indice =
                    i * columnasEsperadas + j;

                float obtenido =
                    actual->datos[indice];

                float esperadoValor =
                    esperado[indice];

                if (!casiIgual(
                        obtenido,
                        esperadoValor,
                        TOLERANCIA))
                {
                    printf("\n[FAIL] %s\n", test);
                    printf("  %s\n", descripcion);

                    printf("  Posicion: [%d][%d]\n",
                           i,
                           j);

                    printf("  Esperado: %.9g\n",
                           esperadoValor);

                    printf("  Obtenido: %.9g\n",
                           obtenido);

                    printf("  Diferencia: %.9g\n",
                           fabsf(
                               obtenido -
                               esperadoValor));

                    fallo = 1;
                }
            }
        }
    }

    if (fallo)
    {
        imprimirMatrizTest(
            "Matriz obtenida",
            actual
        );

        tests_fallados++;
        return 0;
    }

    tests_pasados++;
    return 1;
}


static struct matrix *crearDesdeArray(
    int filas,
    int columnas,
    const float *datos)
{
    struct matrix *m = NULL;

    crearMatriz(
        &m,
        filas,
        columnas
    );

    if (m == NULL)
        return NULL;

    inicializarMatriz(
        m,
        (float *)datos
    );

    return m;
}


static void destruir(struct matrix **m)
{
    if (m != NULL && *m != NULL)
        eliminarMatriz(m);
}


/* ============================================================
 * TESTS DE MSE
 *
 * MSE =
 *
 * (1/N) * SUM((reales - esperadas)^2 / 2)
 *
 * ============================================================ */

static void test_mse_un_elemento(void)
{
    const char *test = "MSE-001";

    float realesDatos[] = {
        3.0f
    };

    float esperadasDatos[] = {
        1.0f
    };

    struct matrix *reales =
        crearDesdeArray(
            1,
            1,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            1,
            1,
            esperadasDatos
        );

    /*
     * N = 1
     *
     * (3 - 1)^2 / 2 = 2
     *
     * 2 / 1 = 2
     */
    float esperado = 2.0f;

    float obtenido =
        mse(
            *reales,
            *esperadas
        );

    if (!casiIgual(
            obtenido,
            esperado,
            TOLERANCIA))
    {
        printf("\n[FAIL] %s - MSE un elemento\n",
               test);

        printf("  reales[0][0] = %.9g\n",
               realesDatos[0]);

        printf("  esperadas[0][0] = %.9g\n",
               esperadasDatos[0]);

        printf("  N = 1\n");

        printf("  Esperado: %.9g\n",
               esperado);

        printf("  Obtenido: %.9g\n",
               obtenido);

        tests_fallados++;
    }
    else
    {
        tests_pasados++;
    }

    destruir(&reales);
    destruir(&esperadas);
}


static void test_mse_igual(void)
{
    const char *test = "MSE-002";

    float realesDatos[] = {
        1.0f,
        2.0f,
        3.0f
    };

    float esperadasDatos[] = {
        1.0f,
        2.0f,
        3.0f
    };

    struct matrix *reales =
        crearDesdeArray(
            3,
            1,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            3,
            1,
            esperadasDatos
        );

    float esperado = 0.0f;

    float obtenido =
        mse(
            *reales,
            *esperadas
        );

    if (!casiIgual(
            obtenido,
            esperado,
            TOLERANCIA))
    {
        printf("\n[FAIL] %s - matrices identicas\n",
               test);

        printf("  N = 3\n");
        printf("  Esperado: %.9g\n",
               esperado);

        printf("  Obtenido: %.9g\n",
               obtenido);

        tests_fallados++;
    }
    else
    {
        tests_pasados++;
    }

    destruir(&reales);
    destruir(&esperadas);
}


static void test_mse_varios_elementos(void)
{
    const char *test = "MSE-003";

    float realesDatos[] = {
        1.0f,
        3.0f,
        5.0f,
        7.0f
    };

    float esperadasDatos[] = {
        0.0f,
        1.0f,
        2.0f,
        3.0f
    };

    /*
     * Diferencias:
     *
     * 1, 2, 3, 4
     *
     * Perdidas:
     *
     * 0.5, 2, 4.5, 8
     *
     * Suma = 15
     *
     * N = 4
     *
     * MSE = 15 / 4 = 3.75
     */
    float esperado = 3.75f;

    struct matrix *reales =
        crearDesdeArray(
            4,
            1,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            4,
            1,
            esperadasDatos
        );

    float obtenido =
        mse(
            *reales,
            *esperadas
        );

    if (!casiIgual(
            obtenido,
            esperado,
            TOLERANCIA))
    {
        printf("\n[FAIL] %s - varios elementos\n",
               test);

        printf("  Suma de perdidas: 15\n");
        printf("  Numero de datos: 4\n");

        printf("  Esperado: %.9g\n",
               esperado);

        printf("  Obtenido: %.9g\n",
               obtenido);

        tests_fallados++;
    }
    else
    {
        tests_pasados++;
    }

    destruir(&reales);
    destruir(&esperadas);
}


static void test_mse_negativos(void)
{
    const char *test = "MSE-004";

    float realesDatos[] = {
        -2.0f,
        -5.0f
    };

    float esperadasDatos[] = {
        1.0f,
        -1.0f
    };

    /*
     * Diferencias:
     *
     * -3, -4
     *
     * Perdidas:
     *
     * 4.5 + 8 = 12.5
     *
     * N = 2
     *
     * MSE = 12.5 / 2 = 6.25
     */
    float esperado = 6.25f;

    struct matrix *reales =
        crearDesdeArray(
            2,
            1,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            2,
            1,
            esperadasDatos
        );

    float obtenido =
        mse(
            *reales,
            *esperadas
        );

    if (!casiIgual(
            obtenido,
            esperado,
            TOLERANCIA))
    {
        printf("\n[FAIL] %s - valores negativos\n",
               test);

        printf("  Suma de perdidas: 12.5\n");
        printf("  Numero de datos: 2\n");

        printf("  Esperado: %.9g\n",
               esperado);

        printf("  Obtenido: %.9g\n",
               obtenido);

        tests_fallados++;
    }
    else
    {
        tests_pasados++;
    }

    destruir(&reales);
    destruir(&esperadas);
}


static void test_mse_2x2(void)
{
    const char *test = "MSE-005";

    float realesDatos[] = {
        1.0f, 2.0f,
        4.0f, 6.0f
    };

    float esperadasDatos[] = {
        0.0f, 1.0f,
        2.0f, 3.0f
    };

    /*
     * Diferencias:
     *
     * 1, 1, 2, 3
     *
     * Perdidas:
     *
     * 0.5 + 0.5 + 2 + 4.5 = 7.5
     *
     * N = 4
     *
     * MSE = 7.5 / 4 = 1.875
     */
    float esperado = 1.875f;

    struct matrix *reales =
        crearDesdeArray(
            2,
            2,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            2,
            2,
            esperadasDatos
        );

    float obtenido =
        mse(
            *reales,
            *esperadas
        );

    if (!casiIgual(
            obtenido,
            esperado,
            TOLERANCIA))
    {
        printf("\n[FAIL] %s - matriz 2x2\n",
               test);

        printf("  Suma de perdidas: 7.5\n");
        printf("  Numero de datos: 4\n");

        printf("  Esperado: %.9g\n",
               esperado);

        printf("  Obtenido: %.9g\n",
               obtenido);

        tests_fallados++;
    }
    else
    {
        tests_pasados++;
    }

    destruir(&reales);
    destruir(&esperadas);
}


static void test_mse_decimales(void)
{
    const char *test = "MSE-006";

    float realesDatos[] = {
        1.5f,
        -2.5f
    };

    float esperadasDatos[] = {
        0.5f,
        -1.0f
    };

    /*
     * Diferencias:
     *
     * 1, -1.5
     *
     * Perdidas:
     *
     * 0.5 + 1.125 = 1.625
     *
     * N = 2
     *
     * MSE = 1.625 / 2 = 0.8125
     */
    float esperado = 0.8125f;

    struct matrix *reales =
        crearDesdeArray(
            2,
            1,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            2,
            1,
            esperadasDatos
        );

    float obtenido =
        mse(
            *reales,
            *esperadas
        );

    if (!casiIgual(
            obtenido,
            esperado,
            TOLERANCIA))
    {
        printf("\n[FAIL] %s - valores decimales\n",
               test);

        printf("  Suma de perdidas: 1.625\n");
        printf("  Numero de datos: 2\n");

        printf("  Esperado: %.9g\n",
               esperado);

        printf("  Obtenido: %.9g\n",
               obtenido);

        tests_fallados++;
    }
    else
    {
        tests_pasados++;
    }

    destruir(&reales);
    destruir(&esperadas);
}


/* ============================================================
 * TESTS DE MSE DERIVADA
 *
 * MSE =
 *
 * (1/N) * SUM((reales - esperadas)^2 / 2)
 *
 * dMSE/dreales =
 *
 * (reales - esperadas) / N
 *
 * ============================================================ */

static void ejecutarTestDerivada(
    const char *test,
    int filas,
    int columnas,
    const float *realesDatos,
    const float *esperadasDatos,
    const float *derivadaEsperada)
{
    struct matrix *reales =
        crearDesdeArray(
            filas,
            columnas,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            filas,
            columnas,
            esperadasDatos
        );

    struct matrix *resultado = NULL;

    crearMatriz(
        &resultado,
        filas,
        columnas
    );

    if (resultado == NULL)
    {
        printf("\n[FAIL] %s\n",
               test);

        printf("  No se pudo crear matriz resultado.\n");

        tests_fallados++;

        destruir(&reales);
        destruir(&esperadas);

        return;
    }

    int retorno =
        mseDerivada(
            *reales,
            *esperadas,
            resultado
        );

    int correcto =
        comprobarMatriz(
            test,
            "Comprobacion de la derivada de MSE",
            resultado,
            filas,
            columnas,
            derivadaEsperada
        );

    if (!correcto)
    {
        printf("  Codigo de retorno de mseDerivada: %d\n",
               retorno);

        imprimirMatrizTest(
            "Entrada reales",
            reales
        );

        imprimirMatrizTest(
            "Entrada esperadas",
            esperadas
        );
    }

    destruir(&reales);
    destruir(&esperadas);
    destruir(&resultado);
}


static void test_mseDerivada_un_elemento(void)
{
    const float reales[] = {
        3.0f
    };

    const float esperadas[] = {
        1.0f
    };

    /*
     * N = 1
     *
     * (3 - 1) / 1 = 2
     */
    const float esperado[] = {
        2.0f
    };

    ejecutarTestDerivada(
        "MSED-001",
        1,
        1,
        reales,
        esperadas,
        esperado
    );
}


static void test_mseDerivada_igual(void)
{
    const float reales[] = {
        1.0f,
        2.0f,
        3.0f
    };

    const float esperadas[] = {
        1.0f,
        2.0f,
        3.0f
    };

    const float esperado[] = {
        0.0f,
        0.0f,
        0.0f
    };

    ejecutarTestDerivada(
        "MSED-002",
        3,
        1,
        reales,
        esperadas,
        esperado
    );
}


static void test_mseDerivada_signos(void)
{
    const float reales[] = {
        1.0f,
        -2.0f,
        5.0f,
        -7.0f
    };

    const float esperadas[] = {
        3.0f,
        -5.0f,
        2.0f,
        -10.0f
    };

    /*
     * Diferencias:
     *
     * -2, 3, 3, 3
     *
     * N = 4
     *
     * Derivada:
     *
     * -0.5, 0.75, 0.75, 0.75
     */
    const float esperado[] = {
        -0.5f,
        0.75f,
        0.75f,
        0.75f
    };

    ejecutarTestDerivada(
        "MSED-003",
        4,
        1,
        reales,
        esperadas,
        esperado
    );
}


static void test_mseDerivada_2x2(void)
{
    const float reales[] = {
        1.0f, 2.0f,
        4.0f, 6.0f
    };

    const float esperadas[] = {
        0.0f, 1.0f,
        2.0f, 3.0f
    };

    /*
     * Diferencias:
     *
     * 1, 1, 2, 3
     *
     * N = 4
     *
     * Derivada:
     *
     * 0.25, 0.25, 0.5, 0.75
     */
    const float esperado[] = {
        0.25f, 0.25f,
        0.5f, 0.75f
    };

    ejecutarTestDerivada(
        "MSED-004",
        2,
        2,
        reales,
        esperadas,
        esperado
    );
}


static void test_mseDerivada_decimales(void)
{
    const float reales[] = {
        1.5f,
        -2.5f
    };

    const float esperadas[] = {
        0.5f,
        -1.0f
    };

    /*
     * Diferencias:
     *
     * 1, -1.5
     *
     * N = 2
     *
     * Derivada:
     *
     * 0.5, -0.75
     */
    const float esperado[] = {
        0.5f,
        -0.75f
    };

    ejecutarTestDerivada(
        "MSED-005",
        2,
        1,
        reales,
        esperadas,
        esperado
    );
}


/* ============================================================
 * COMPROBACION NUMERICA DE LA DERIVADA
 * ============================================================ */

static void test_consistencia_mse_derivada(void)
{
    const char *test = "MSED-006";

    const int filas = 3;
    const int columnas = 1;
    const int numeroDatos = filas * columnas;

    const float realesDatos[] = {
        2.0f,
        -1.5f,
        4.25f
    };

    const float esperadasDatos[] = {
        0.5f,
        2.0f,
        1.25f
    };

    struct matrix *reales =
        crearDesdeArray(
            filas,
            columnas,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            filas,
            columnas,
            esperadasDatos
        );

    struct matrix *derivada = NULL;

    crearMatriz(
        &derivada,
        filas,
        columnas
    );

    if (reales == NULL ||
        esperadas == NULL ||
        derivada == NULL)
    {
        printf("\n[FAIL] %s\n",
               test);

        printf("  No se pudieron crear las matrices necesarias.\n");

        tests_fallados++;

        destruir(&reales);
        destruir(&esperadas);
        destruir(&derivada);

        return;
    }

    int retorno =
        mseDerivada(
            *reales,
            *esperadas,
            derivada
        );

    (void)retorno;

    const float epsilon = 0.0001f;

    int fallo = 0;

    for (int i = 0; i < filas; i++)
    {
        float original =
            reales->datos[i];

        reales->datos[i] =
            original + epsilon;

        float superior =
            mse(
                *reales,
                *esperadas
            );

        reales->datos[i] =
            original - epsilon;

        float inferior =
            mse(
                *reales,
                *esperadas
            );

        reales->datos[i] =
            original;

        float derivadaNumerica =
            (superior - inferior) /
            (2.0f * epsilon);

        float derivadaAnalitica =
            derivada->datos[i];

        if (!casiIgual(
                derivadaAnalitica,
                derivadaNumerica,
                TOLERANCIA_NUMERICA))
        {
            printf("\n[FAIL] %s\n",
                   test);

            printf("  Elemento: [%d][0]\n",
                   i);

            printf("  reales original: %.9g\n",
                   original);

            printf("  esperadas: %.9g\n",
                   esperadas->datos[i]);

            printf("  Numero de datos N: %d\n",
                   numeroDatos);

            printf("  epsilon: %.9g\n",
                   epsilon);

            printf("  Derivada analitica: %.9g\n",
                   derivadaAnalitica);

            printf("  Derivada numerica: %.9g\n",
                   derivadaNumerica);

            printf("  Diferencia: %.9g\n",
                   fabsf(
                       derivadaAnalitica -
                       derivadaNumerica));

            fallo = 1;
        }
    }

    if (fallo)
        tests_fallados++;
    else
        tests_pasados++;

    destruir(&reales);
    destruir(&esperadas);
    destruir(&derivada);
}


/* ============================================================
 * TEST DE NO MODIFICACION DE ENTRADAS
 * ============================================================ */

static void test_mse_no_modifica_entradas(void)
{
    const char *test = "MSE-007";

    const float realesDatos[] = {
        1.0f,
        -2.0f,
        3.5f
    };

    const float esperadasDatos[] = {
        0.5f,
        -1.0f,
        4.0f
    };

    struct matrix *reales =
        crearDesdeArray(
            3,
            1,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            3,
            1,
            esperadasDatos
        );

    if (reales == NULL ||
        esperadas == NULL)
    {
        printf("\n[FAIL] %s\n",
               test);

        printf("  No se pudieron crear las matrices.\n");

        tests_fallados++;

        destruir(&reales);
        destruir(&esperadas);

        return;
    }

    float realesAntes[3];
    float esperadasAntes[3];

    for (int i = 0; i < 3; i++)
    {
        realesAntes[i] =
            reales->datos[i];

        esperadasAntes[i] =
            esperadas->datos[i];
    }

    (void)mse(
        *reales,
        *esperadas
    );

    int fallo = 0;

    for (int i = 0; i < 3; i++)
    {
        if (reales->datos[i] !=
            realesAntes[i])
        {
            fallo = 1;
        }

        if (esperadas->datos[i] !=
            esperadasAntes[i])
        {
            fallo = 1;
        }
    }

    if (fallo)
    {
        printf("\n[FAIL] %s\n",
               test);

        printf("  mse() ha modificado alguna matriz de entrada.\n");

        printf("  Reales antes:     ");

        for (int i = 0; i < 3; i++)
            printf("%.9g ",
                   realesAntes[i]);

        printf("\n  Reales despues:   ");

        for (int i = 0; i < 3; i++)
            printf("%.9g ",
                   reales->datos[i]);

        printf("\n  Esperadas antes:  ");

        for (int i = 0; i < 3; i++)
            printf("%.9g ",
                   esperadasAntes[i]);

        printf("\n  Esperadas despues:");

        for (int i = 0; i < 3; i++)
            printf(" %.9g",
                   esperadas->datos[i]);

        printf("\n");

        tests_fallados++;
    }
    else
    {
        tests_pasados++;
    }

    destruir(&reales);
    destruir(&esperadas);
}


/* ============================================================
 * MATRIZ GRANDE
 * ============================================================ */

static void test_matriz_grande(void)
{
    const char *test = "MSE-008 / MSED-008";

    const int filas = 5;
    const int columnas = 4;
    const int numeroDatos = filas * columnas;

    float realesDatos[20];
    float esperadasDatos[20];
    float derivadaEsperada[20];

    float sumaPerdidas = 0.0f;

    for (int i = 0; i < numeroDatos; i++)
    {
        realesDatos[i] =
            (float)(i - 7) * 0.5f;

        esperadasDatos[i] =
            (float)(i % 5 - 2) * 0.75f;

        float diferencia =
            realesDatos[i] -
            esperadasDatos[i];

        derivadaEsperada[i] =
            diferencia /
            (float)numeroDatos;

        sumaPerdidas +=
            diferencia *
            diferencia /
            2.0f;
    }

    float esperadoMSE =
        sumaPerdidas /
        (float)numeroDatos;

    struct matrix *reales =
        crearDesdeArray(
            filas,
            columnas,
            realesDatos
        );

    struct matrix *esperadas =
        crearDesdeArray(
            filas,
            columnas,
            esperadasDatos
        );

    struct matrix *derivada = NULL;

    crearMatriz(
        &derivada,
        filas,
        columnas
    );

    if (reales == NULL ||
        esperadas == NULL ||
        derivada == NULL)
    {
        printf("\n[FAIL] %s\n",
               test);

        printf("  Error creando matrices.\n");

        tests_fallados++;

        destruir(&reales);
        destruir(&esperadas);
        destruir(&derivada);

        return;
    }

    float obtenidoMSE =
        mse(
            *reales,
            *esperadas
        );

    if (!casiIgual(
            obtenidoMSE,
            esperadoMSE,
            TOLERANCIA))
    {
        printf("\n[FAIL] %s - MSE matriz 5x4\n",
               test);

        printf("  Numero de datos: %d\n",
               numeroDatos);

        printf("  Suma de perdidas: %.9g\n",
               sumaPerdidas);

        printf("  Esperado: %.9g\n",
               esperadoMSE);

        printf("  Obtenido: %.9g\n",
               obtenidoMSE);

        tests_fallados++;
    }
    else
    {
        tests_pasados++;
    }

    int retorno =
        mseDerivada(
            *reales,
            *esperadas,
            derivada
        );

    (void)retorno;

    int falloDerivada = 0;

    for (int i = 0; i < numeroDatos; i++)
    {
        if (!casiIgual(
                derivada->datos[i],
                derivadaEsperada[i],
                TOLERANCIA))
        {
            printf(
                "\n[FAIL] %s - derivada posicion %d\n",
                test,
                i
            );

            printf("  Esperado: %.9g\n",
                   derivadaEsperada[i]);

            printf("  Obtenido: %.9g\n",
                   derivada->datos[i]);

            falloDerivada = 1;
        }
    }

    if (falloDerivada)
        tests_fallados++;
    else
        tests_pasados++;

    destruir(&reales);
    destruir(&esperadas);
    destruir(&derivada);
}


/* ============================================================
 * MAIN
 * ============================================================ */

int main(void)
{
    printf("=============================================\n");
    printf("             TEST LOSS - MSE\n");
    printf("=============================================\n");

    test_mse_un_elemento();
    test_mse_igual();
    test_mse_varios_elementos();
    test_mse_negativos();
    test_mse_2x2();
    test_mse_decimales();

    test_mseDerivada_un_elemento();
    test_mseDerivada_igual();
    test_mseDerivada_signos();
    test_mseDerivada_2x2();
    test_mseDerivada_decimales();

    test_consistencia_mse_derivada();

    test_mse_no_modifica_entradas();

    test_matriz_grande();

    printf("\n=============================================\n");
    printf("                  RESUMEN\n");
    printf("=============================================\n");

    printf("Tests pasados : %d\n",
           tests_pasados);

    printf("Tests fallados: %d\n",
           tests_fallados);

    printf("=============================================\n");

    if (tests_fallados == 0)
    {
        printf("TODOS LOS TESTS HAN PASADO.\n");
        return EXIT_SUCCESS;
    }

    printf("HAY TESTS FALLIDOS.\n");
    return EXIT_FAILURE;
}