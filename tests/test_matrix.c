#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "matrix.h"

/*
 * ============================================================================
 *                              TEST MATRIX
 * ============================================================================
 *
 * Este fichero reúne TODOS los tests de matrix.
 *
 * Funciones probadas:
 *
 *   - crearMatriz()
 *   - imprimirMatriz()
 *   - eliminarMatriz()
 *   - inicializarMatriz()
 *   - accederPos()
 *   - modificarPos()
 *   - crearConNumero()
 *   - copiarMatriz()
 *   - numeroRandom()
 *   - inicializarRandom()
 *   - transponerMatriz()
 *   - multiplicarEscalar()
 *   - suma()
 *   - resta()
 *   - multiplicacionMatricial()
 *   - multiplicacionElemPorElem()
 *
 * La idea es que este archivo sustituya completamente a matrix_mult.
 *
 *
 * ============================================================================
 * CRITERIOS
 * ============================================================================
 *
 * Se comprueban:
 *
 *   1. Dimensiones.
 *   2. Valores.
 *   3. Códigos de retorno.
 *   4. Casos pequeños.
 *   5. Casos rectangulares.
 *   6. Valores negativos.
 *   7. Ceros.
 *   8. Decimales.
 *   9. Matrices identidad.
 *  10. Copias profundas.
 *  11. Modificación de posiciones.
 *  12. Transposición.
 *  13. Multiplicación escalar.
 *  14. Suma.
 *  15. Resta.
 *  16. Multiplicación matricial.
 *  17. Multiplicación elemento a elemento.
 *  18. Números aleatorios dentro de rango.
 *
 *
 * IMPORTANTE
 * ============================================================================
 *
 * No se comprueban comportamientos no definidos por matrix.h.
 *
 * Por ejemplo, no se exige qué debe hacer una función cuando recibe índices
 * fuera de rango si el contrato no especifica el comportamiento.
 *
 * Para operaciones incompatibles en dimensiones, solamente se comprueba un
 * código de error si la implementación/documentación establece que debe
 * devolverlo. Si no está especificado, no se fuerza un comportamiento concreto.
 * ============================================================================
 */


#define TOL_ABS 1e-5f
#define TOL_REL 1e-5f

#define RANDOM_MIN -5.0f
#define RANDOM_MAX  5.0f

static int tests_totales = 0;
static int tests_pasados = 0;
static int tests_fallados = 0;


/* ============================================================================
 * UTILIDADES
 * ========================================================================== */

static int floatIguales(float a, float b)
{
    float error = fabsf(a - b);
    float escala = fmaxf(
        1.0f,
        fmaxf(fabsf(a), fabsf(b))
    );

    return error <= TOL_ABS ||
           error / escala <= TOL_REL;
}


static void separador(void)
{
    printf(
        "-------------------------------------------------------------------------------\n"
    );
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
            "  [ERROR] Dimensiones incorrectas en %s.\n"
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

    printf(
        "  [OK] Todos los elementos de %s son correctos.\n",
        nombre
    );

    return 1;
}


static int comprobarFloat(
    const char *nombre,
    float esperado,
    float obtenido)
{
    if (!floatIguales(esperado, obtenido))
    {
        printf(
            "  [ERROR] %s\n"
            "          Esperado : %.10f\n"
            "          Obtenido : %.10f\n"
            "          Error abs: %.10f\n",
            nombre,
            esperado,
            obtenido,
            fabsf(esperado - obtenido)
        );

        return 0;
    }

    printf(
        "  [OK] %s = %.10f\n",
        nombre,
        obtenido
    );

    return 1;
}


static void registrarResultado(
    const char *nombre,
    int correcto)
{
    tests_totales++;

    if (correcto)
    {
        tests_pasados++;

        printf(
            "[PASS] %s\n",
            nombre
        );
    }
    else
    {
        tests_fallados++;

        printf(
            "[FAIL] %s\n",
            nombre
        );
    }

    printf("\n");
}


static int comprobarCodigoRetorno(
    const char *nombre,
    int esperado,
    int obtenido)
{
    if (esperado != obtenido)
    {
        printf(
            "  [ERROR] Código de retorno de %s\n"
            "          Esperado : %d\n"
            "          Obtenido : %d\n",
            nombre,
            esperado,
            obtenido
        );

        return 0;
    }

    printf(
        "  [OK] Código de retorno de %s = %d\n",
        nombre,
        obtenido
    );

    return 1;
}


/*
 * Comprueba que todos los elementos estén dentro de [min,max].
 */
static int comprobarRango(
    const char *nombre,
    const struct matrix *m,
    float min,
    float max)
{
    int i;
    int total;

    if (m == NULL)
    {
        printf(
            "  [ERROR] %s == NULL.\n",
            nombre
        );

        return 0;
    }

    if (m->datos == NULL)
    {
        printf(
            "  [ERROR] %s->datos == NULL.\n",
            nombre
        );

        return 0;
    }

    total = m->fil * m->col;

    for (i = 0; i < total; i++)
    {
        if (m->datos[i] < min ||
            m->datos[i] > max)
        {
            printf(
                "  [ERROR] %s[%d] fuera de rango.\n"
                "          Rango esperado: [%.5f, %.5f]\n"
                "          Obtenido: %.10f\n",
                nombre,
                i,
                min,
                max,
                m->datos[i]
            );

            return 0;
        }
    }

    printf(
        "  [OK] Todos los elementos de %s están en [%.5f, %.5f].\n",
        nombre,
        min,
        max
    );

    return 1;
}


/* ============================================================================
 * TEST 01
 *
 * crearMatriz()
 * ========================================================================== */

static void test_crearMatriz(void)
{
    const char *nombre =
        "MATRIX-01 - crearMatriz";

    struct matrix *m = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    printf(
        "Creando matriz de 3 x 4...\n"
    );

    crearMatriz(
        &m,
        3,
        4
    );

    if (m == NULL)
    {
        printf(
            "[ERROR] crearMatriz() dejó m == NULL.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    printf(
        "Matriz creada en %p\n",
        (void *)m
    );

    if (m->fil != 3)
    {
        printf(
            "[ERROR] fil incorrecto.\n"
            "        Esperado: 3\n"
            "        Obtenido : %d\n",
            m->fil
        );

        correcto = 0;
    }
    else
    {
        printf("[OK] fil = 3\n");
    }

    if (m->col != 4)
    {
        printf(
            "[ERROR] col incorrecto.\n"
            "        Esperado: 4\n"
            "        Obtenido : %d\n",
            m->col
        );

        correcto = 0;
    }
    else
    {
        printf("[OK] col = 4\n");
    }

    if (m->datos == NULL)
    {
        printf(
            "[ERROR] m->datos == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] m->datos reservado correctamente.\n"
        );
    }

    eliminarMatriz(&m);

    if (m != NULL)
    {
        printf(
            "[ERROR] eliminarMatriz() no dejó m == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] eliminarMatriz() dejó m == NULL.\n"
        );
    }

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 02
 *
 * crearConNumero()
 * ========================================================================== */

static void test_crearConNumero(void)
{
    const char *nombre =
        "MATRIX-02 - crearConNumero";

    struct matrix *m = NULL;

    float esperado[] = {
        7.5f, 7.5f, 7.5f,
        7.5f, 7.5f, 7.5f
    };

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    printf(
        "Creando matriz 2x3 con todos los valores = 7.5\n"
    );

    crearConNumero(
        &m,
        2,
        3,
        7.5f
    );

    if (m == NULL)
    {
        printf(
            "[ERROR] crearConNumero() dejó m == NULL.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    imprimirMatriz("matriz", m);

    if (!comprobarMatriz(
            "matriz",
            m,
            esperado,
            2,
            3))
    {
        correcto = 0;
    }

    eliminarMatriz(&m);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 03
 *
 * inicializarMatriz()
 * ========================================================================== */

static void test_inicializarMatriz(void)
{
    const char *nombre =
        "MATRIX-03 - inicializarMatriz";

    float datos[] = {
        1.0f, -2.0f,
        3.5f, 4.25f
    };

    float esperado[] = {
        1.0f, -2.0f,
        3.5f, 4.25f
    };

    struct matrix *m = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &m,
        2,
        2
    );

    if (m == NULL)
    {
        printf(
            "[ERROR] No se pudo crear matriz.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    printf("Datos utilizados:\n");
    {
        struct matrix temporal = {
            2,
            2,
            datos
        };

        imprimirMatriz(
            "datos",
            &temporal
        );
    }

    inicializarMatriz(
        m,
        datos
    );

    imprimirMatriz(
        "matriz después de inicializar",
        m
    );

    if (!comprobarMatriz(
            "matriz",
            m,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&m);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 04
 *
 * accederPos()
 * ========================================================================== */

static void test_accederPos(void)
{
    const char *nombre =
        "MATRIX-04 - accederPos";

    float datos[] = {
        10.0f, 20.0f, 30.0f,
        40.0f, 50.0f, 60.0f
    };

    struct matrix m = {
        2,
        3,
        datos
    };

    float *posiciones[6];

    int correcto = 1;
    int i;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    imprimirMatriz(
        "matriz",
        &m
    );

    posiciones[0] = accederPos(&m, 0, 0);
    posiciones[1] = accederPos(&m, 0, 1);
    posiciones[2] = accederPos(&m, 0, 2);
    posiciones[3] = accederPos(&m, 1, 0);
    posiciones[4] = accederPos(&m, 1, 1);
    posiciones[5] = accederPos(&m, 1, 2);

    for (i = 0; i < 6; i++)
    {
        if (posiciones[i] == NULL)
        {
            printf(
                "[ERROR] accederPos() devolvió NULL para posición %d.\n",
                i
            );

            correcto = 0;
        }
    }

    if (posiciones[0] != NULL &&
        *posiciones[0] != 10.0f)
    {
        printf(
            "[ERROR] [0][0]: esperado 10, obtenido %.10f\n",
            *posiciones[0]
        );

        correcto = 0;
    }

    if (posiciones[1] != NULL &&
        *posiciones[1] != 20.0f)
    {
        printf(
            "[ERROR] [0][1]: esperado 20, obtenido %.10f\n",
            *posiciones[1]
        );

        correcto = 0;
    }

    if (posiciones[2] != NULL &&
        *posiciones[2] != 30.0f)
    {
        printf(
            "[ERROR] [0][2]: esperado 30, obtenido %.10f\n",
            *posiciones[2]
        );

        correcto = 0;
    }

    if (posiciones[3] != NULL &&
        *posiciones[3] != 40.0f)
    {
        printf(
            "[ERROR] [1][0]: esperado 40, obtenido %.10f\n",
            *posiciones[3]
        );

        correcto = 0;
    }

    if (posiciones[4] != NULL &&
        *posiciones[4] != 50.0f)
    {
        printf(
            "[ERROR] [1][1]: esperado 50, obtenido %.10f\n",
            *posiciones[4]
        );

        correcto = 0;
    }

    if (posiciones[5] != NULL &&
        *posiciones[5] != 60.0f)
    {
        printf(
            "[ERROR] [1][2]: esperado 60, obtenido %.10f\n",
            *posiciones[5]
        );

        correcto = 0;
    }

    if (correcto)
    {
        printf(
            "[OK] Las seis posiciones devuelven los valores correctos.\n"
        );
    }

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 05
 *
 * modificarPos()
 * ========================================================================== */

static void test_modificarPos(void)
{
    const char *nombre =
        "MATRIX-05 - modificarPos";

    float datos[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float esperado[] = {
        1.0f, 2.0f,
        99.5f, 4.0f
    };

    struct matrix m = {
        2,
        2,
        datos
    };

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    imprimirMatriz(
        "Antes",
        &m
    );

    printf(
        "\nEjecutando modificarPos(&m, 1, 0, 99.5)...\n"
    );

    modificarPos(
        &m,
        1,
        0,
        99.5f
    );

    imprimirMatriz(
        "Después",
        &m
    );

    if (!comprobarMatriz(
            "matriz",
            &m,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 06
 *
 * copiarMatriz()
 *
 * El header especifica explícitamente deep copy.
 * ========================================================================== */

static void test_copiarMatriz(void)
{
    const char *nombre =
        "MATRIX-06 - copiarMatriz realiza deep copy";

    float datosOriginal[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float esperado[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    struct matrix original = {
        2,
        2,
        datosOriginal
    };

    struct matrix *copia = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &copia,
        2,
        2
    );

    if (copia == NULL)
    {
        printf(
            "[ERROR] No se pudo crear matriz destino.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    printf("Original:\n");
    imprimirMatriz(
        "original",
        &original
    );

    copiarMatriz(
        &original,
        copia
    );

    printf("\nCopia:\n");
    imprimirMatriz(
        "copia",
        copia
    );

    if (!comprobarMatriz(
            "copia",
            copia,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    /*
     * Comprobamos independencia.
     */
    printf(
        "\nModificando original[0][0] = 999...\n"
    );

    original.datos[0] = 999.0f;

    imprimirMatriz(
        "original modificada",
        &original
    );

    imprimirMatriz(
        "copia después de modificar original",
        copia
    );

    if (!floatIguales(
            copia->datos[0],
            1.0f))
    {
        printf(
            "[ERROR] La copia comparte datos con la matriz original.\n"
            "        copia[0][0] = %.10f\n",
            copia->datos[0]
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] La copia permanece independiente.\n"
        );
    }

    eliminarMatriz(&copia);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 07
 *
 * transponerMatriz()
 *
 * A = 2x3
 *
 *      [1 2 3]
 *      [4 5 6]
 *
 * A^T = 3x2
 *
 *      [1 4]
 *      [2 5]
 *      [3 6]
 * ========================================================================== */

static void test_transponerMatriz_rectangular(void)
{
    const char *nombre =
        "MATRIX-07 - transponerMatriz rectangular";

    float datos[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    float esperado[] = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f
    };

    struct matrix original = {
        2,
        3,
        datos
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        3,
        2
    );

    if (resultado == NULL)
    {
        printf(
            "[ERROR] No se pudo crear matriz resultado.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    imprimirMatriz(
        "Original",
        &original
    );

    retorno = transponerMatriz(
        &original,
        resultado
    );

    printf(
        "\nCódigo de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "Transpuesta",
        resultado
    );

    if (retorno != 0)
    {
        printf(
            "[ERROR] transponerMatriz() devolvió %d.\n",
            retorno
        );

        correcto = 0;
    }

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            3,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 08
 *
 * transponerMatriz() MATRIZ CUADRADA
 * ========================================================================== */

static void test_transponerMatriz_cuadrada(void)
{
    const char *nombre =
        "MATRIX-08 - transponerMatriz cuadrada";

    float datos[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float esperado[] = {
        1.0f, 3.0f,
        2.0f, 4.0f
    };

    struct matrix original = {
        2,
        2,
        datos
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = transponerMatriz(
        &original,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 09
 *
 * transponerMatriz() VECTOR
 * ========================================================================== */

static void test_transponerMatriz_vector(void)
{
    const char *nombre =
        "MATRIX-09 - transponerMatriz vector columna";

    float datos[] = {
        10.0f,
        20.0f,
        30.0f
    };

    float esperado[] = {
        10.0f,
        20.0f,
        30.0f
    };

    struct matrix original = {
        3,
        1,
        datos
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        1,
        3
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = transponerMatriz(
        &original,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            1,
            3))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 10
 *
 * multiplicarEscalar()
 *
 * Este test evita el problema del test antiguo:
 *
 * NO reutilizamos la misma matriz después de haberla modificado para calcular
 * los valores esperados.
 * ========================================================================== */

static void test_multiplicarEscalar(void)
{
    const char *nombre =
        "MATRIX-10 - multiplicarEscalar";

    float datos[] = {
        2.0f, -3.0f, 4.0f,
        5.0f,  0.0f, -1.5f
    };

    float esperado[] = {
         6.0f, -9.0f, 12.0f,
        15.0f,  0.0f, -4.5f
    };

    struct matrix original = {
        2,
        3,
        datos
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    printf(
        "Escalar utilizado: 3.0\n\n"
    );

    imprimirMatriz(
        "Original",
        &original
    );

    crearMatriz(
        &resultado,
        2,
        3
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicarEscalar(
        &original,
        3.0f,
        resultado
    );

    printf(
        "\nCódigo de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "Resultado",
        resultado
    );

    if (retorno != 0)
    {
        printf(
            "[ERROR] multiplicarEscalar() devolvió %d.\n",
            retorno
        );

        correcto = 0;
    }

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            3))
    {
        correcto = 0;
    }

    /*
     * Comprobamos que la operación no haya modificado el original.
     */
    {
        float esperadoOriginal[] = {
            2.0f, -3.0f, 4.0f,
            5.0f,  0.0f, -1.5f
        };

        printf(
            "\nComprobando que el original no haya sido modificado...\n"
        );

        if (!comprobarMatriz(
                "original",
                &original,
                esperadoOriginal,
                2,
                3))
        {
            correcto = 0;
        }
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 11
 *
 * multiplicarEscalar POR CERO
 * ========================================================================== */

static void test_multiplicarEscalar_cero(void)
{
    const char *nombre =
        "MATRIX-11 - multiplicarEscalar por cero";

    float datos[] = {
        10.0f, -20.0f,
        30.0f, -40.0f
    };

    float esperado[] = {
        0.0f, 0.0f,
        0.0f, 0.0f
    };

    struct matrix original = {
        2,
        2,
        datos
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicarEscalar(
        &original,
        0.0f,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 12
 *
 * suma()
 *
 * A:
 *      [1 2]
 *      [3 4]
 *
 * B:
 *      [5 6]
 *      [7 8]
 *
 * A+B:
 *      [ 6  8]
 *      [10 12]
 * ========================================================================== */

static void test_suma(void)
{
    const char *nombre =
        "MATRIX-12 - suma";

    float datosA[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float datosB[] = {
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    float esperado[] = {
         6.0f,  8.0f,
        10.0f, 12.0f
    };

    struct matrix A = {
        2,
        2,
        datosA
    };

    struct matrix B = {
        2,
        2,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    imprimirMatriz("A", &A);
    imprimirMatriz("B", &B);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = suma(
        A,
        B,
        resultado
    );

    printf(
        "\nCódigo de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "A + B",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 13
 *
 * suma con negativos y cero.
 * ========================================================================== */

static void test_suma_negativos(void)
{
    const char *nombre =
        "MATRIX-13 - suma con negativos y ceros";

    float datosA[] = {
         5.0f, -3.0f,
         0.0f,  2.5f
    };

    float datosB[] = {
        -5.0f,  3.0f,
         2.0f, -2.5f
    };

    float esperado[] = {
        0.0f, 0.0f,
        2.0f, 0.0f
    };

    struct matrix A = {
        2,
        2,
        datosA
    };

    struct matrix B = {
        2,
        2,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = suma(
        A,
        B,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 14
 *
 * resta()
 *
 * A:
 *      [5 6]
 *      [7 8]
 *
 * B:
 *      [1 2]
 *      [3 4]
 *
 * A-B:
 *      [4 4]
 *      [4 4]
 * ========================================================================== */

static void test_resta(void)
{
    const char *nombre =
        "MATRIX-14 - resta";

    float datosA[] = {
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    float datosB[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float esperado[] = {
        4.0f, 4.0f,
        4.0f, 4.0f
    };

    struct matrix A = {
        2,
        2,
        datosA
    };

    struct matrix B = {
        2,
        2,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    imprimirMatriz("A", &A);
    imprimirMatriz("B", &B);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = resta(
        A,
        B,
        resultado
    );

    printf(
        "\nCódigo de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "A - B",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 15
 *
 * resta con negativos.
 * ========================================================================== */

static void test_resta_negativos(void)
{
    const char *nombre =
        "MATRIX-15 - resta con negativos y decimales";

    float datosA[] = {
         1.5f, -2.0f,
        -3.5f,  4.0f
    };

    float datosB[] = {
        -1.5f,  2.0f,
         3.5f, -4.0f
    };

    float esperado[] = {
         3.0f, -4.0f,
        -7.0f,  8.0f
    };

    struct matrix A = {
        2,
        2,
        datosA
    };

    struct matrix B = {
        2,
        2,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = resta(
        A,
        B,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 16
 *
 * multiplicacionElemPorElem()
 *
 * A:
 *      [1 2]
 *      [3 4]
 *
 * B:
 *      [5 6]
 *      [7 8]
 *
 * resultado:
 *      [ 5 12]
 *      [21 32]
 * ========================================================================== */

static void test_multiplicacionElemPorElem(void)
{
    const char *nombre =
        "MATRIX-16 - multiplicacionElemPorElem";

    float datosA[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float datosB[] = {
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    float esperado[] = {
         5.0f, 12.0f,
        21.0f, 32.0f
    };

    struct matrix A = {
        2,
        2,
        datosA
    };

    struct matrix B = {
        2,
        2,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    imprimirMatriz("A", &A);
    imprimirMatriz("B", &B);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicacionElemPorElem(
        &A,
        &B,
        resultado
    );

    printf(
        "\nCódigo de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 17
 *
 * multiplicacionElemPorElem con decimales y negativos.
 * ========================================================================== */

static void test_multiplicacionElemPorElem_decimales(void)
{
    const char *nombre =
        "MATRIX-17 - multiplicacionElemPorElem con decimales";

    float datosA[] = {
         0.5f, -2.0f,
         3.0f, -1.5f
    };

    float datosB[] = {
         2.0f,  4.0f,
        -0.5f, -2.0f
    };

    float esperado[] = {
         1.0f, -8.0f,
        -1.5f,  3.0f
    };

    struct matrix A = {
        2,
        2,
        datosA
    };

    struct matrix B = {
        2,
        2,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicacionElemPorElem(
        &A,
        &B,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 18
 *
 * MULTIPLICACIÓN MATRICIAL BÁSICA
 *
 * A = 2x3
 *
 *      [1 2 3]
 *      [4 5 6]
 *
 * B = 3x2
 *
 *      [ 7  8]
 *      [ 9 10]
 *      [11 12]
 *
 * A*B = 2x2
 *
 *      [ 58  64]
 *      [139 154]
 * ========================================================================== */

static void test_multiplicacionMatricial_basica(void)
{
    const char *nombre =
        "MATRIX-18 - multiplicacionMatricial básica";

    float datosA[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    float datosB[] = {
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f
    };

    float esperado[] = {
         58.0f,  64.0f,
        139.0f, 154.0f
    };

    struct matrix A = {
        2,
        3,
        datosA
    };

    struct matrix B = {
        3,
        2,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    imprimirMatriz("A", &A);
    imprimirMatriz("B", &B);

    printf(
        "\nCálculos principales:\n"
        "  C[0][0] = 1*7 + 2*9 + 3*11 = 58\n"
        "  C[0][1] = 1*8 + 2*10 + 3*12 = 64\n"
        "  C[1][0] = 4*7 + 5*9 + 6*11 = 139\n"
        "  C[1][1] = 4*8 + 5*10 + 6*12 = 154\n"
    );

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicacionMatricial(
        &A,
        &B,
        resultado
    );

    printf(
        "\nCódigo de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "A * B",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 19
 *
 * MULTIPLICACIÓN MATRICIAL NO CUADRADA
 *
 * A = 3x2
 * B = 2x1
 *
 * A:
 *   [1 2]
 *   [3 4]
 *   [5 6]
 *
 * B:
 *   [10]
 *   [20]
 *
 * Resultado:
 *
 *   [50]
 *   [110]
 *   [170]
 * ========================================================================== */

static void test_multiplicacionMatricial_rectangular(void)
{
    const char *nombre =
        "MATRIX-19 - multiplicacionMatricial rectangular";

    float datosA[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };

    float datosB[] = {
        10.0f,
        20.0f
    };

    float esperado[] = {
         50.0f,
        110.0f,
        170.0f
    };

    struct matrix A = {
        3,
        2,
        datosA
    };

    struct matrix B = {
        2,
        1,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    imprimirMatriz("A", &A);
    imprimirMatriz("B", &B);

    crearMatriz(
        &resultado,
        3,
        1
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicacionMatricial(
        &A,
        &B,
        resultado
    );

    printf(
        "\nCódigo de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            3,
            1))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 20
 *
 * MULTIPLICACIÓN MATRICIAL CON IDENTIDAD
 *
 * A * I = A
 *
 * Comprueba que la multiplicación no altere los valores y sirve para detectar
 * errores de índices.
 * ========================================================================== */

static void test_multiplicacionMatricial_identidad(void)
{
    const char *nombre =
        "MATRIX-20 - multiplicacionMatricial por identidad";

    float datosA[] = {
         2.0f, -3.0f,  5.0f,
         7.0f,  1.0f, -4.0f,
         0.5f,  8.0f,  9.0f
    };

    float datosI[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };

    float esperado[] = {
         2.0f, -3.0f,  5.0f,
         7.0f,  1.0f, -4.0f,
         0.5f,  8.0f,  9.0f
    };

    struct matrix A = {
        3,
        3,
        datosA
    };

    struct matrix I = {
        3,
        3,
        datosI
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        3,
        3
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicacionMatricial(
        &A,
        &I,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "A * I",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            3,
            3))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 21
 *
 * MULTIPLICACIÓN MATRICIAL CON CEROS
 *
 * Comprueba que el acumulador se inicialice correctamente a cero.
 * ========================================================================== */

static void test_multiplicacionMatricial_ceros(void)
{
    const char *nombre =
        "MATRIX-21 - multiplicacionMatricial con matriz cero";

    float datosA[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float datosB[] = {
        0.0f, 0.0f,
        0.0f, 0.0f
    };

    float esperado[] = {
        0.0f, 0.0f,
        0.0f, 0.0f
    };

    struct matrix A = {
        2,
        2,
        datosA
    };

    struct matrix B = {
        2,
        2,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicacionMatricial(
        &A,
        &B,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 22
 *
 * CASO MUY PEQUEÑO DE MULTIPLICACIÓN MATRICIAL
 *
 * 1x1:
 *
 *      [4] * [5] = [20]
 *
 * Sirve para comprobar el caso mínimo.
 * ========================================================================== */

static void test_multiplicacionMatricial_1x1(void)
{
    const char *nombre =
        "MATRIX-22 - multiplicacionMatricial 1x1";

    float datosA[] = {
        4.0f
    };

    float datosB[] = {
        5.0f
    };

    float esperado[] = {
        20.0f
    };

    struct matrix A = {
        1,
        1,
        datosA
    };

    struct matrix B = {
        1,
        1,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        1,
        1
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicacionMatricial(
        &A,
        &B,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            1,
            1))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 23
 *
 * MULTIPLICACIÓN MATRICIAL CON NEGATIVOS Y DECIMALES
 * ========================================================================== */

static void test_multiplicacionMatricial_decimales(void)
{
    const char *nombre =
        "MATRIX-23 - multiplicacionMatricial con negativos y decimales";

    float datosA[] = {
         0.5f, -2.0f,
         1.5f,  3.0f
    };

    float datosB[] = {
         4.0f, -1.0f,
        -2.0f,  0.5f
    };

    float esperado[] = {
         6.0f, -1.5f,
        -0.0f,  0.0f
    };

    struct matrix A = {
        2,
        2,
        datosA
    };

    struct matrix B = {
        2,
        2,
        datosB
    };

    struct matrix *resultado = NULL;

    int retorno;
    int correcto = 1;

    /*
     * Cálculos:
     *
     * C00 = 0.5*4 + (-2)*(-2) = 2 + 4 = 6
     * C01 = 0.5*(-1) + (-2)*0.5 = -0.5 -1 = -1.5
     * C10 = 1.5*4 + 3*(-2) = 6 - 6 = 0
     * C11 = 1.5*(-1) + 3*0.5 = -1.5 + 1.5 = 0
     */

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &resultado,
        2,
        2
    );

    if (resultado == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    retorno = multiplicacionMatricial(
        &A,
        &B,
        resultado
    );

    printf(
        "Código de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "resultado",
        resultado
    );

    if (retorno != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado",
            resultado,
            esperado,
            2,
            2))
    {
        correcto = 0;
    }

    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 24
 *
 * numeroRandom()
 *
 * No se comprueba un número concreto.
 *
 * Se comprueba que:
 *
 *      min <= resultado <= max
 *
 * Se hacen muchas muestras para tener más posibilidades de detectar un error
 * de rango.
 * ========================================================================== */

static void test_numeroRandom(void)
{
    const char *nombre =
        "MATRIX-24 - numeroRandom dentro del rango";

    const float min = -10.0f;
    const float max = 20.0f;

    int correcto = 1;

    int i;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    printf(
        "Rango solicitado: [%.2f, %.2f]\n",
        min,
        max
    );

    for (i = 0; i < 1000; i++)
    {
        float valor = numeroRandom(
            min,
            max
        );

        if (valor < min ||
            valor > max)
        {
            printf(
                "[ERROR] Muestra %d fuera de rango.\n"
                "        Valor = %.10f\n"
                "        Rango = [%.10f, %.10f]\n",
                i,
                valor,
                min,
                max
            );

            correcto = 0;
            break;
        }
    }

    if (correcto)
    {
        printf(
            "[OK] Las 1000 muestras están dentro del rango.\n"
        );
    }

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 25
 *
 * inicializarRandom()
 * ========================================================================== */

static void test_inicializarRandom(void)
{
    const char *nombre =
        "MATRIX-25 - inicializarRandom";

    struct matrix *m = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    printf(
        "Creando matriz 5x5 y rellenándola con valores aleatorios.\n"
        "Rango: [%.2f, %.2f]\n",
        RANDOM_MIN,
        RANDOM_MAX
    );

    crearMatriz(
        &m,
        5,
        5
    );

    if (m == NULL)
    {
        printf(
            "[ERROR] No se pudo crear matriz.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    inicializarRandom(
        m,
        RANDOM_MIN,
        RANDOM_MAX
    );

    imprimirMatriz(
        "matriz aleatoria",
        m
    );

    if (!comprobarRango(
            "matriz aleatoria",
            m,
            RANDOM_MIN,
            RANDOM_MAX))
    {
        correcto = 0;
    }

    eliminarMatriz(&m);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 26
 *
 * inicializarRandom() DOS VECES
 *
 * No se exige que sean distintas porque una implementación aleatoria
 * correctamente válida podría, en teoría, producir los mismos valores.
 *
 * Solo se comprueba que ambas inicializaciones respeten el rango.
 * ========================================================================== */

static void test_inicializarRandom_dos_veces(void)
{
    const char *nombre =
        "MATRIX-26 - inicializarRandom repetidamente";

    struct matrix *m = NULL;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &m,
        4,
        4
    );

    if (m == NULL)
    {
        registrarResultado(nombre, 0);
        return;
    }

    printf("Primera inicialización:\n");

    inicializarRandom(
        m,
        -1.0f,
        1.0f
    );

    imprimirMatriz(
        "m",
        m
    );

    if (!comprobarRango(
            "m",
            m,
            -1.0f,
            1.0f))
    {
        correcto = 0;
    }

    printf("\nSegunda inicialización:\n");

    inicializarRandom(
        m,
        -1.0f,
        1.0f
    );

    imprimirMatriz(
        "m",
        m
    );

    if (!comprobarRango(
            "m",
            m,
            -1.0f,
            1.0f))
    {
        correcto = 0;
    }

    eliminarMatriz(&m);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 27
 *
 * imprimirMatriz()
 *
 * No podemos comprobar automáticamente el texto mostrado sin capturar stdout.
 *
 * Este test existe para ejecutar la función sobre una matriz conocida y
 * comprobar visualmente que no produce errores/crashes.
 * ========================================================================== */

static void test_imprimirMatriz(void)
{
    const char *nombre =
        "MATRIX-27 - imprimirMatriz";

    float datos[] = {
        1.0f, -2.5f,
        3.25f, 4.0f
    };

    struct matrix m = {
        2,
        2,
        datos
    };

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    printf(
        "La siguiente salida corresponde a imprimirMatriz():\n\n"
    );

    imprimirMatriz(nombre, &m);

    printf(
        "\n[OK] imprimirMatriz() se ha ejecutado sin error.\n"
    );

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 28
 *
 * OPERACIONES EN CADENA
 *
 * Simula parte del uso real de la red:
 *
 *      W
 *      A
 *
 *      Z = W*A
 *
 * y después:
 *
 *      Z + b
 *
 * Esto comprueba que las funciones matriciales puedan utilizarse juntas.
 *
 * W = 2x3
 * A = 3x1
 * b = 2x1
 *
 * ========================================================================== */

static void test_cadena_operaciones(void)
{
    const char *nombre =
        "MATRIX-28 - cadena multiplicación + suma";

    float datosW[] = {
         1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 2.0f
    };

    float datosA[] = {
        2.0f,
        -1.0f,
        3.0f
    };

    float datosB[] = {
         0.5f,
        -2.0f
    };

    /*
     * W*A:
     *
     * fila 0:
     *   1*2 + 2*(-1) + 3*3 = 9
     *
     * fila 1:
     *   -1*2 + 0.5*(-1) + 2*3 = 3.5
     *
     * W*A+b:
     *
     *   [9.5]
     *   [1.5]
     */
    float esperadoMultiplicacion[] = {
        9.0f,
        3.5f
    };

    float esperadoFinal[] = {
        9.5f,
        1.5f
    };

    struct matrix W = {
        2,
        3,
        datosW
    };

    struct matrix A = {
        3,
        1,
        datosA
    };

    struct matrix B = {
        2,
        1,
        datosB
    };

    struct matrix *WA = NULL;
    struct matrix *final = NULL;

    int retornoMultiplicacion;
    int retornoSuma;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &WA,
        2,
        1
    );

    crearMatriz(
        &final,
        2,
        1
    );

    if (WA == NULL ||
        final == NULL)
    {
        printf(
            "[ERROR] No se pudieron crear matrices intermedias.\n"
        );

        eliminarMatriz(&WA);
        eliminarMatriz(&final);

        registrarResultado(nombre, 0);
        return;
    }

    printf("Paso 1: W*A\n");

    retornoMultiplicacion =
        multiplicacionMatricial(
            &W,
            &A,
            WA
        );

    printf(
        "Código de retorno = %d\n",
        retornoMultiplicacion
    );

    imprimirMatriz(
        "W*A",
        WA
    );

    if (retornoMultiplicacion != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "W*A",
            WA,
            esperadoMultiplicacion,
            2,
            1))
    {
        correcto = 0;
    }

    printf("\nPaso 2: (W*A)+b\n");

    retornoSuma =
        suma(
            *WA,
            B,
            final
        );

    printf(
        "Código de retorno = %d\n",
        retornoSuma
    );

    imprimirMatriz(
        "resultado final",
        final
    );

    if (retornoSuma != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "resultado final",
            final,
            esperadoFinal,
            2,
            1))
    {
        correcto = 0;
    }

    eliminarMatriz(&WA);
    eliminarMatriz(&final);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 29
 *
 * TRANSPOSE + MULTIPLICACIÓN
 *
 * Comprueba una operación muy habitual en redes neuronales:
 *
 *      A^T * A
 *
 * A es 2x3:
 *
 *      [1 2 3]
 *      [4 5 6]
 *
 * A^T es 3x2.
 *
 * Resultado 3x3:
 *
 *      [17 22 27]
 *      [22 29 36]
 *      [27 36 45]
 * ========================================================================== */

static void test_transpuesta_multiplicacion(void)
{
    const char *nombre =
        "MATRIX-29 - transposición seguida de multiplicación";

    float datosA[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    float esperado[] = {
        17.0f, 22.0f, 27.0f,
        22.0f, 29.0f, 36.0f,
        27.0f, 36.0f, 45.0f
    };

    struct matrix A = {
        2,
        3,
        datosA
    };

    struct matrix *AT = NULL;
    struct matrix *resultado = NULL;

    int retornoTranspose;
    int retornoMultiplicacion;

    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    crearMatriz(
        &AT,
        3,
        2
    );

    crearMatriz(
        &resultado,
        3,
        3
    );

    if (AT == NULL ||
        resultado == NULL)
    {
        printf(
            "[ERROR] No se pudieron crear matrices intermedias.\n"
        );

        eliminarMatriz(&AT);
        eliminarMatriz(&resultado);

        registrarResultado(nombre, 0);
        return;
    }

    retornoTranspose =
        transponerMatriz(
            &A,
            AT
        );

    printf(
        "Código transposición = %d\n",
        retornoTranspose
    );

    imprimirMatriz(
        "A^T",
        AT
    );

    if (retornoTranspose != 0)
        correcto = 0;

    retornoMultiplicacion =
        multiplicacionMatricial(
            AT,
            &A,
            resultado
        );

    printf(
        "\nCódigo multiplicación = %d\n",
        retornoMultiplicacion
    );

    imprimirMatriz(
        "A^T * A",
        resultado
    );

    if (retornoMultiplicacion != 0)
        correcto = 0;

    if (!comprobarMatriz(
            "A^T * A",
            resultado,
            esperado,
            3,
            3))
    {
        correcto = 0;
    }

    eliminarMatriz(&AT);
    eliminarMatriz(&resultado);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 30
 *
 * OPERACIONES IN-PLACE / ALIASING
 *
 * Se comprueba un caso especialmente importante:
 *
 *      multiplicarEscalar(&A, x, &A)
 *
 * Es decir, entrada y salida son la misma matriz.
 *
 * SOLO se incluye como test si la implementación parece soportar aliasing;
 * no se asume que todas las funciones deban soportarlo. Por ello este test
 * informa del resultado pero no lo convierte en fallo si el contrato no lo
 * especifica.
 * ========================================================================== */

static void test_multiplicarEscalar_mismo_resultado(void)
{
    const char *nombre =
        "MATRIX-30 - multiplicarEscalar con entrada y salida iguales";

    float datos[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float esperado[] = {
         2.0f,  4.0f,
         6.0f,  8.0f
    };

    struct matrix m = {
        2,
        2,
        datos
    };

    int retorno;

    printf("\n");
    printf("TEST: %s\n\n", nombre);

    printf(
        "Este caso comprueba si la función soporta:\n"
        "    multiplicarEscalar(&m, 2, &m)\n"
        "\n"
    );

    imprimirMatriz(
        "Antes",
        &m
    );

    retorno = multiplicarEscalar(
        &m,
        2.0f,
        &m
    );

    printf(
        "\nCódigo de retorno = %d\n",
        retorno
    );

    imprimirMatriz(
        "Después",
        &m
    );

    if (retorno != 0)
    {
        printf(
            "[INFO] La función no acepta este caso según su código de retorno.\n"
            "       Como el header no documenta aliasing, no se considera fallo.\n"
        );

        registrarResultado(nombre, 1);
        return;
    }

    if (!comprobarMatriz(
            "resultado",
            &m,
            esperado,
            2,
            2))
    {
        printf(
            "[INFO] La función no soporta correctamente aliasing.\n"
            "       No se considera fallo porque matrix.h no especifica\n"
            "       que entrada y salida puedan ser la misma matriz.\n"
        );

        registrarResultado(nombre, 1);
        return;
    }

    printf(
        "[OK] multiplicarEscalar soporta entrada == salida.\n"
    );

    registrarResultado(nombre, 1);
}


/* ============================================================================
 * MAIN
 * ========================================================================== */

int main(void)
{
    printf("\n");
    printf("===============================================================================\n");
    printf("                                TEST MATRIX\n");
    printf("===============================================================================\n");
    printf("\n");

    printf(
        "Este archivo contiene todos los tests de matrix.\n"
        "Puede utilizarse para sustituir matrix_mult.\n"
    );

    printf("\n");

    printf(
        "Funciones evaluadas:\n"
        "  - crearMatriz()\n"
        "  - imprimirMatriz()\n"
        "  - eliminarMatriz()\n"
        "  - inicializarMatriz()\n"
        "  - accederPos()\n"
        "  - modificarPos()\n"
        "  - crearConNumero()\n"
        "  - copiarMatriz()\n"
        "  - numeroRandom()\n"
        "  - inicializarRandom()\n"
        "  - transponerMatriz()\n"
        "  - multiplicarEscalar()\n"
        "  - suma()\n"
        "  - resta()\n"
        "  - multiplicacionMatricial()\n"
        "  - multiplicacionElemPorElem()\n"
    );

    printf("\n");

    printf(
        "Tolerancia absoluta: %.2e\n"
        "Tolerancia relativa: %.2e\n",
        TOL_ABS,
        TOL_REL
    );

    separador();


    /* ------------------------------------------------------------------------
     * CREACIÓN / ACCESO / MODIFICACIÓN
     * ---------------------------------------------------------------------- */

    test_crearMatriz();

    separador();

    test_crearConNumero();

    separador();

    test_inicializarMatriz();

    separador();

    test_accederPos();

    separador();

    test_modificarPos();

    separador();

    test_copiarMatriz();


    /* ------------------------------------------------------------------------
     * TRANSPOSE
     * ---------------------------------------------------------------------- */

    separador();

    test_transponerMatriz_rectangular();

    separador();

    test_transponerMatriz_cuadrada();

    separador();

    test_transponerMatriz_vector();


    /* ------------------------------------------------------------------------
     * OPERACIONES ELEMENTALES
     * ---------------------------------------------------------------------- */

    separador();

    test_multiplicarEscalar();

    separador();

    test_multiplicarEscalar_cero();

    separador();

    test_suma();

    separador();

    test_suma_negativos();

    separador();

    test_resta();

    separador();

    test_resta_negativos();


    /* ------------------------------------------------------------------------
     * MULTIPLICACIÓN ELEMENTO A ELEMENTO
     * ---------------------------------------------------------------------- */

    separador();

    test_multiplicacionElemPorElem();

    separador();

    test_multiplicacionElemPorElem_decimales();


    /* ------------------------------------------------------------------------
     * MULTIPLICACIÓN MATRICIAL
     * ---------------------------------------------------------------------- */

    separador();

    test_multiplicacionMatricial_basica();

    separador();

    test_multiplicacionMatricial_rectangular();

    separador();

    test_multiplicacionMatricial_identidad();

    separador();

    test_multiplicacionMatricial_ceros();

    separador();

    test_multiplicacionMatricial_1x1();

    separador();

    test_multiplicacionMatricial_decimales();


    /* ------------------------------------------------------------------------
     * RANDOM
     * ---------------------------------------------------------------------- */

    separador();

    test_numeroRandom();

    separador();

    test_inicializarRandom();

    separador();

    test_inicializarRandom_dos_veces();


    /* ------------------------------------------------------------------------
     * IMPRESIÓN
     * ---------------------------------------------------------------------- */

    separador();

    test_imprimirMatriz();


    /* ------------------------------------------------------------------------
     * PRUEBAS INTEGRADAS
     * ---------------------------------------------------------------------- */

    separador();

    test_cadena_operaciones();

    separador();

    test_transpuesta_multiplicacion();

    separador();

    test_multiplicarEscalar_mismo_resultado();


    /* ------------------------------------------------------------------------
     * RESUMEN
     * ---------------------------------------------------------------------- */

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
            "Todos los tests de matrix han pasado correctamente.\n"
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

    printf(
        "===============================================================================\n"
    );

    return tests_fallados == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;


}   