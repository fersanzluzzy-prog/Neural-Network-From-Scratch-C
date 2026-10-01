#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "layer.h"
#include "matrix.h"

/*
 * ============================================================================
 * TEST CACHE
 * ============================================================================
 *
 * Funciones probadas:
 *
 *   - inicializarCache()
 *   - guardarCacheLayerA()
 *   - guardarCacheLayerZ()
 *   - eliminarCacheLayer()
 *
 * NO se prueba guardarCacheLayer()
 * --------------------------------
 * Esa función está declarada en layer.h, pero no está implementada en el
 * proyecto y, por tanto, no debe utilizarse en los tests.
 *
 * Estructura:
 *
 *      struct cache {
 *          struct matrix *a;
 *          struct matrix *z;
 *      };
 *
 * El cache debe poder almacenar independientemente:
 *
 *      A -> activación de la capa
 *      Z -> preactivación de la capa
 *
 * Se comprueba:
 *
 *   1. Inicialización correcta.
 *   2. Punteros inicialmente NULL.
 *   3. Guardado de A.
 *   4. Guardado de Z.
 *   5. Sustitución de A.
 *   6. Sustitución de Z.
 *   7. Independencia entre A y Z.
 *   8. Matrices de distintos tamaños.
 *   9. Valores negativos, cero y positivos.
 *  10. Deep copy / comportamiento de memoria según la implementación.
 *  11. Eliminación correcta.
 *
 * IMPORTANTE SOBRE MEMORIA
 * ------------------------
 * guardarCacheLayerA() y guardarCacheLayerZ() reciben una matriz por valor:
 *
 *      void guardarCacheLayerA(struct cache *cache, struct matrix a1);
 *      void guardarCacheLayerZ(struct cache *cache, struct matrix z1);
 *
 * Por tanto, la función recibe la estructura matrix por valor, pero la matriz
 * contiene un puntero a datos.
 *
 * Este test comprueba que el contenido almacenado sea correcto y, además,
 * comprueba si el cache mantiene el contenido después de modificar la matriz
 * original. Esto permite detectar si se está almacenando solamente el puntero
 * en lugar de hacer una copia independiente.
 *
 * Si la implementación está diseñada deliberadamente para compartir memoria,
 * ese test deberá interpretarse según el contrato de la asignatura/proyecto.
 * ========================================================================== */


#define TOL_ABS 1e-6f
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
           (error / escala) <= TOL_REL;
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
}


static int crearMatrizValores(
    struct matrix **resultado,
    int filas,
    int columnas,
    const float *datos)
{
    int i;

    *resultado = NULL;

    crearMatriz(resultado, filas, columnas);

    if (*resultado == NULL)
        return 0;

    if ((*resultado)->datos == NULL)
        return 0;

    for (i = 0; i < filas * columnas; i++)
        (*resultado)->datos[i] = datos[i];

    return 1;
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
        printf("  [ERROR] %s == NULL\n", nombre);
        return 0;
    }

    if (obtenida->fil != filas ||
        obtenida->col != columnas)
    {
        printf(
            "  [ERROR] %s tiene dimensiones incorrectas.\n"
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
        printf("  [ERROR] %s->datos == NULL\n", nombre);
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
            "  [ERROR] %s tiene %d valor(es) incorrecto(s).\n",
            nombre,
            errores
        );

        return 0;
    }

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
 * TEST 1
 *
 * Inicialización.
 *
 * Se espera:
 *
 *      cache != NULL
 *      cache->a == NULL
 *      cache->z == NULL
 * ========================================================================== */

static void test_inicializarCache(void)
{
    const char *nombre =
        "CACHE-01 - inicializarCache crea un cache vacío";

    struct cache *cache;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf(
            "[ERROR] inicializarCache() devolvió NULL.\n"
            "        No se puede utilizar el cache.\n"
        );

        registrarResultado(nombre, 0);
        return;
    }

    printf("Dirección del cache: %p\n", (void *)cache);
    printf("cache->a: %p\n", (void *)cache->a);
    printf("cache->z: %p\n", (void *)cache->z);

    if (cache->a != NULL)
    {
        printf(
            "[ERROR] Un cache recién inicializado tiene a != NULL.\n"
            "        Se esperaba a == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        printf("[OK] cache->a está correctamente inicializado a NULL.\n");
    }

    if (cache->z != NULL)
    {
        printf(
            "[ERROR] Un cache recién inicializado tiene z != NULL.\n"
            "        Se esperaba z == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        printf("[OK] cache->z está correctamente inicializado a NULL.\n");
    }

    eliminarCacheLayer(&cache);

    printf(
        "Cache eliminado mediante eliminarCacheLayer().\n"
    );

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 2
 *
 * Guardar A.
 * ========================================================================== */

static void test_guardarA(void)
{
    const char *nombre =
        "CACHE-02 - guardarCacheLayerA almacena correctamente A";

    float datos[] = {
         1.0f,  2.0f,
        -3.0f,  4.5f
    };

    struct matrix entrada;
    struct cache *cache;

    int correcto = 1;

    entrada.fil = 2;
    entrada.col = 2;
    entrada.datos = datos;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear el cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    imprimirMatriz("A que se va a guardar", &entrada);

    guardarCacheLayerA(cache, entrada);

    printf("\nEstado del cache después de guardar A:\n");
    imprimirMatriz("cache->a", cache->a);

    if (cache->a == NULL)
    {
        printf(
            "[ERROR] guardarCacheLayerA() dejó cache->a == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        if (!comprobarMatriz(
                "cache->a",
                cache->a,
                datos,
                2,
                2))
        {
            correcto = 0;
        }
        else
        {
            printf(
                "[OK] A se ha almacenado con dimensiones y valores correctos.\n"
            );
        }
    }

    if (cache->z != NULL)
    {
        printf(
            "[ERROR] Guardar A también modificó cache->z.\n"
            "        cache->z debería seguir siendo NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] cache->z sigue siendo NULL; A y Z son independientes.\n"
        );
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 3
 *
 * Guardar Z.
 * ========================================================================== */

static void test_guardarZ(void)
{
    const char *nombre =
        "CACHE-03 - guardarCacheLayerZ almacena correctamente Z";

    float datos[] = {
        -1.5f,  0.0f,  3.25f
    };

    struct matrix entrada;
    struct cache *cache;

    int correcto = 1;

    entrada.fil = 1;
    entrada.col = 3;
    entrada.datos = datos;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear el cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    imprimirMatriz("Z que se va a guardar", &entrada);

    guardarCacheLayerZ(cache, entrada);

    printf("\nEstado del cache después de guardar Z:\n");
    imprimirMatriz("cache->z", cache->z);

    if (cache->z == NULL)
    {
        printf(
            "[ERROR] guardarCacheLayerZ() dejó cache->z == NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        if (!comprobarMatriz(
                "cache->z",
                cache->z,
                datos,
                1,
                3))
        {
            correcto = 0;
        }
        else
        {
            printf(
                "[OK] Z se ha almacenado con dimensiones y valores correctos.\n"
            );
        }
    }

    if (cache->a != NULL)
    {
        printf(
            "[ERROR] Guardar Z también modificó cache->a.\n"
            "        cache->a debería seguir siendo NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] cache->a sigue siendo NULL; Z y A son independientes.\n"
        );
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 4
 *
 * Guardar A y Z en el mismo cache.
 *
 * Es fundamental porque una capa necesita conservar ambos valores:
 *
 *      Z = W*A_anterior + b
 *      A = f(Z)
 *
 * El cache de la capa almacena la A producida y el Z de esa misma capa.
 * ========================================================================== */

static void test_guardarA_y_Z(void)
{
    const char *nombre =
        "CACHE-04 - guardar A y Z simultáneamente";

    float datosA[] = {
         0.25f,
        -0.75f
    };

    float datosZ[] = {
         1.20f,
        -2.40f
    };

    struct matrix matrizA;
    struct matrix matrizZ;

    struct cache *cache;

    int correcto = 1;

    matrizA.fil = 2;
    matrizA.col = 1;
    matrizA.datos = datosA;

    matrizZ.fil = 2;
    matrizZ.col = 1;
    matrizZ.datos = datosZ;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear el cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    imprimirMatriz("A original", &matrizA);
    imprimirMatriz("Z original", &matrizZ);

    guardarCacheLayerA(cache, matrizA);
    guardarCacheLayerZ(cache, matrizZ);

    printf("\nEstado final del cache:\n");
    imprimirMatriz("cache->a", cache->a);
    imprimirMatriz("cache->z", cache->z);

    if (!comprobarMatriz(
            "cache->a",
            cache->a,
            datosA,
            2,
            1))
    {
        correcto = 0;
    }

    if (!comprobarMatriz(
            "cache->z",
            cache->z,
            datosZ,
            2,
            1))
    {
        correcto = 0;
    }

    if (cache->a != NULL && cache->z != NULL)
    {
        if (cache->a == cache->z)
        {
            printf(
                "[ERROR] cache->a y cache->z apuntan a la misma estructura.\n"
                "        A y Z deben almacenarse independientemente.\n"
            );

            correcto = 0;
        }
        else
        {
            printf(
                "[OK] cache->a y cache->z son estructuras independientes.\n"
            );
        }
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 5
 *
 * Sustitución de A.
 *
 * guardarCacheLayerA() debe permitir actualizar la activación almacenada.
 * ========================================================================== */

static void test_sustituirA(void)
{
    const char *nombre =
        "CACHE-05 - sustituir una A previamente almacenada";

    float datosA1[] = {
         1.0f,
         2.0f
    };

    float datosA2[] = {
        -10.0f,
         7.5f
    };

    struct matrix A1;
    struct matrix A2;

    struct cache *cache;

    int correcto = 1;

    A1.fil = 2;
    A1.col = 1;
    A1.datos = datosA1;

    A2.fil = 2;
    A2.col = 1;
    A2.datos = datosA2;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("Primera A:\n");
    imprimirMatriz("A1", &A1);

    guardarCacheLayerA(cache, A1);

    printf("\nCache después de A1:\n");
    imprimirMatriz("cache->a", cache->a);

    if (!comprobarMatriz(
            "cache->a",
            cache->a,
            datosA1,
            2,
            1))
    {
        correcto = 0;
    }

    printf("\nSegunda A:\n");
    imprimirMatriz("A2", &A2);

    guardarCacheLayerA(cache, A2);

    printf("\nCache después de sustituir A1 por A2:\n");
    imprimirMatriz("cache->a", cache->a);

    if (!comprobarMatriz(
            "cache->a",
            cache->a,
            datosA2,
            2,
            1))
    {
        correcto = 0;
    }
    else
    {
        printf(
            "[OK] La activación almacenada contiene A2.\n"
        );
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 6
 *
 * Sustitución de Z.
 * ========================================================================== */

static void test_sustituirZ(void)
{
    const char *nombre =
        "CACHE-06 - sustituir una Z previamente almacenada";

    float datosZ1[] = {
         1.0f,
         2.0f,
         3.0f
    };

    float datosZ2[] = {
        -5.0f,
         0.0f,
         8.0f
    };

    struct matrix Z1;
    struct matrix Z2;

    struct cache *cache;

    int correcto = 1;

    Z1.fil = 3;
    Z1.col = 1;
    Z1.datos = datosZ1;

    Z2.fil = 3;
    Z2.col = 1;
    Z2.datos = datosZ2;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    guardarCacheLayerZ(cache, Z1);

    printf("Cache después de Z1:\n");
    imprimirMatriz("cache->z", cache->z);

    guardarCacheLayerZ(cache, Z2);

    printf("\nCache después de sustituir Z1 por Z2:\n");
    imprimirMatriz("cache->z", cache->z);

    if (!comprobarMatriz(
            "cache->z",
            cache->z,
            datosZ2,
            3,
            1))
    {
        correcto = 0;
    }
    else
    {
        printf(
            "[OK] La preactivación almacenada contiene Z2.\n"
        );
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 7
 *
 * Comprobar independencia entre A y Z.
 *
 * Guardamos A, después Z y finalmente volvemos a comprobar A.
 * Esto detecta errores donde guardar Z pisa accidentalmente A.
 * ========================================================================== */

static void test_independencia_A_Z(void)
{
    const char *nombre =
        "CACHE-07 - A y Z no se sobrescriben entre sí";

    float datosA[] = {
         10.0f,
         20.0f
    };

    float datosZ[] = {
        -30.0f,
        -40.0f
    };

    struct matrix A;
    struct matrix Z;

    struct cache *cache;

    int correcto = 1;

    A.fil = 2;
    A.col = 1;
    A.datos = datosA;

    Z.fil = 2;
    Z.col = 1;
    Z.datos = datosZ;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    guardarCacheLayerA(cache, A);

    printf("Después de guardar A:\n");
    imprimirMatriz("cache->a", cache->a);
    imprimirMatriz("cache->z", cache->z);

    guardarCacheLayerZ(cache, Z);

    printf("\nDespués de guardar Z:\n");
    imprimirMatriz("cache->a", cache->a);
    imprimirMatriz("cache->z", cache->z);

    if (!comprobarMatriz(
            "cache->a",
            cache->a,
            datosA,
            2,
            1))
    {
        printf(
            "  [ERROR] Guardar Z ha alterado A.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "  [OK] A sigue intacta después de guardar Z.\n"
        );
    }

    if (!comprobarMatriz(
            "cache->z",
            cache->z,
            datosZ,
            2,
            1))
    {
        correcto = 0;
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 8
 *
 * Deep copy de A.
 *
 * Guardamos una matriz, modificamos posteriormente la matriz original y
 * comprobamos si el cache conserva el valor anterior.
 *
 * Este test es importante para detectar:
 *
 *      cache->a = &matriz
 *
 * frente a una copia real de la matriz.
 *
 * Se espera una copia independiente porque el cache necesita conservar los
 * valores de forward aunque las matrices temporales cambien posteriormente.
 * ========================================================================== */

static void test_copia_independiente_A(void)
{
    const char *nombre =
        "CACHE-08 - A almacenada de forma independiente de la entrada";

    float datos[] = {
         1.0f,
         2.0f,
         3.0f,
         4.0f
    };

    float esperadoOriginal[] = {
         1.0f,
         2.0f,
         3.0f,
         4.0f
    };

    struct matrix entrada;
    struct cache *cache;

    int correcto = 1;

    entrada.fil = 2;
    entrada.col = 2;
    entrada.datos = datos;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf("Matriz antes de guardar:\n");
    imprimirMatriz("entrada", &entrada);

    guardarCacheLayerA(cache, entrada);

    printf("\nContenido almacenado inicialmente:\n");
    imprimirMatriz("cache->a", cache->a);

    /*
     * Modificar la matriz original DESPUÉS de guardar.
     */
    datos[0] = 100.0f;
    datos[1] = 200.0f;
    datos[2] = 300.0f;
    datos[3] = 400.0f;

    printf("\nMatriz original después de modificarla:\n");
    imprimirMatriz("entrada", &entrada);

    printf("\nContenido que debería conservar el cache:\n");
    imprimirMatriz("cache->a", cache->a);

    if (!comprobarMatriz(
            "cache->a",
            cache->a,
            esperadoOriginal,
            2,
            2))
    {
        printf(
            "[ERROR] cache->a cambió al modificar la matriz original.\n"
            "        Esto indica que el cache probablemente comparte\n"
            "        el mismo buffer de datos.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] cache->a conserva los valores originales.\n"
            "     La matriz almacenada es independiente de la entrada.\n"
        );
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 9
 *
 * Deep copy de Z.
 * ========================================================================== */

static void test_copia_independiente_Z(void)
{
    const char *nombre =
        "CACHE-09 - Z almacenada de forma independiente de la entrada";

    float datos[] = {
        -1.0f,
        -2.0f,
        -3.0f
    };

    float esperadoOriginal[] = {
        -1.0f,
        -2.0f,
        -3.0f
    };

    struct matrix entrada;
    struct cache *cache;

    int correcto = 1;

    entrada.fil = 3;
    entrada.col = 1;
    entrada.datos = datos;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    guardarCacheLayerZ(cache, entrada);

    printf("Z almacenada inicialmente:\n");
    imprimirMatriz("cache->z", cache->z);

    /*
     * Modificar entrada después de guardar.
     */
    datos[0] = 100.0f;
    datos[1] = 200.0f;
    datos[2] = 300.0f;

    printf("\nZ original después de modificarla:\n");
    imprimirMatriz("entrada", &entrada);

    printf("\nContenido del cache:\n");
    imprimirMatriz("cache->z", cache->z);

    if (!comprobarMatriz(
            "cache->z",
            cache->z,
            esperadoOriginal,
            3,
            1))
    {
        printf(
            "[ERROR] cache->z cambió al modificar la matriz original.\n"
            "        Esto indica que probablemente comparte el buffer.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] cache->z conserva los valores originales.\n"
            "     La matriz almacenada es independiente de la entrada.\n"
        );
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 10
 *
 * Matrices con diferentes dimensiones.
 *
 * Comprueba que no se haya implementado el cache suponiendo siempre vectores
 * columna de un tamaño concreto.
 * ========================================================================== */

static void test_dimensiones_variadas(void)
{
    const char *nombre =
        "CACHE-10 - matrices de diferentes dimensiones";

    float datosA[] = {
         1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f
    };

    float datosZ[] = {
        -1.0f, -2.0f,
        -3.0f, -4.0f,
        -5.0f, -6.0f,
        -7.0f, -8.0f
    };

    struct matrix A;
    struct matrix Z;

    struct cache *cache;

    int correcto = 1;

    A.fil = 2;
    A.col = 3;
    A.datos = datosA;

    Z.fil = 4;
    Z.col = 2;
    Z.datos = datosZ;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    imprimirMatriz("A", &A);
    imprimirMatriz("Z", &Z);

    guardarCacheLayerA(cache, A);
    guardarCacheLayerZ(cache, Z);

    printf("\nCache resultante:\n");
    imprimirMatriz("cache->a", cache->a);
    imprimirMatriz("cache->z", cache->z);

    if (!comprobarMatriz(
            "cache->a",
            cache->a,
            datosA,
            2,
            3))
    {
        correcto = 0;
    }

    if (!comprobarMatriz(
            "cache->z",
            cache->z,
            datosZ,
            4,
            2))
    {
        correcto = 0;
    }

    if (cache->a != NULL &&
        cache->z != NULL &&
        cache->a->fil == 2 &&
        cache->a->col == 3 &&
        cache->z->fil == 4 &&
        cache->z->col == 2)
    {
        printf(
            "[OK] A y Z conservan sus dimensiones originales.\n"
        );
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 11
 *
 * Valores especiales dentro de matrices normales:
 *
 *      negativos
 *      cero
 *      positivos
 *      valores decimales
 *
 * No se utilizan NaN ni infinito porque no forman parte del comportamiento
 * normal esperado de una caché de una red neuronal.
 * ========================================================================== */

static void test_valores_variados(void)
{
    const char *nombre =
        "CACHE-11 - valores negativos, cero y decimales";

    float datosA[] = {
        -100.5f,
           0.0f,
         0.0001f,
         999.999f
    };

    float datosZ[] = {
         50.25f,
        -0.0002f,
          0.0f,
        -750.75f
    };

    struct matrix A;
    struct matrix Z;

    struct cache *cache;

    int correcto = 1;

    A.fil = 4;
    A.col = 1;
    A.datos = datosA;

    Z.fil = 4;
    Z.col = 1;
    Z.datos = datosZ;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    guardarCacheLayerA(cache, A);
    guardarCacheLayerZ(cache, Z);

    imprimirMatriz("A original", &A);
    imprimirMatriz("cache->a", cache->a);

    imprimirMatriz("Z original", &Z);
    imprimirMatriz("cache->z", cache->z);

    if (!comprobarMatriz(
            "cache->a",
            cache->a,
            datosA,
            4,
            1))
    {
        correcto = 0;
    }

    if (!comprobarMatriz(
            "cache->z",
            cache->z,
            datosZ,
            4,
            1))
    {
        correcto = 0;
    }

    eliminarCacheLayer(&cache);

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 12
 *
 * Eliminación.
 *
 * Se comprueba:
 *
 *      eliminarCacheLayer(&cache)
 *
 * y posteriormente que el puntero haya quedado en NULL.
 *
 * Esto depende de la convención habitual utilizada por el proyecto:
 *
 *      void eliminarX(struct X **x)
 *
 * donde la función recibe un puntero al puntero para poder dejarlo a NULL.
 * ========================================================================== */

static void test_eliminarCache(void)
{
    const char *nombre =
        "CACHE-12 - eliminarCacheLayer libera y anula el puntero";

    float datosA[] = {
        1.0f,
        2.0f
    };

    float datosZ[] = {
        3.0f,
        4.0f
    };

    struct matrix A;
    struct matrix Z;

    struct cache *cache;

    int correcto = 1;

    A.fil = 2;
    A.col = 1;
    A.datos = datosA;

    Z.fil = 2;
    Z.col = 1;
    Z.datos = datosZ;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] No se pudo crear cache.\n");
        registrarResultado(nombre, 0);
        return;
    }

    guardarCacheLayerA(cache, A);
    guardarCacheLayerZ(cache, Z);

    printf("Antes de eliminar:\n");
    printf("  cache = %p\n", (void *)cache);
    printf("  cache->a = %p\n", (void *)cache->a);
    printf("  cache->z = %p\n", (void *)cache->z);

    printf("\nEjecutando eliminarCacheLayer(&cache)...\n");

    {
        int retorno = eliminarCacheLayer(&cache);

        printf(
            "Código de retorno de eliminarCacheLayer(): %d\n",
            retorno
        );

        /*
         * La función devuelve int, por lo que se registra el código.
         * No se impone aquí un valor concreto porque el header no documenta
         * explícitamente cuál es el código de éxito.
         */
    }

    printf("\nDespués de eliminar:\n");
    printf("  cache = %p\n", (void *)cache);

    if (cache != NULL)
    {
        printf(
            "[ERROR] eliminarCacheLayer() no dejó cache == NULL.\n"
            "        Se esperaba que el puntero recibido por referencia\n"
            "        quedase invalidado después de liberar el objeto.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] cache == NULL después de eliminarCacheLayer().\n"
        );
    }

    registrarResultado(nombre, correcto);
}


/* ============================================================================
 * TEST 13
 *
 * Eliminación de un cache vacío.
 *
 * Comprueba que no sea necesario haber almacenado A o Z antes de eliminar.
 * ========================================================================== */

static void test_eliminarCacheVacio(void)
{
    const char *nombre =
        "CACHE-13 - eliminar un cache recién inicializado";

    struct cache *cache;
    int correcto = 1;

    printf("\n");
    printf("TEST: %s\n", nombre);
    printf("\n");

    cache = inicializarCache();

    if (cache == NULL)
    {
        printf("[ERROR] inicializarCache() devolvió NULL.\n");
        registrarResultado(nombre, 0);
        return;
    }

    printf(
        "Cache creado: %p\n",
        (void *)cache
    );

    printf(
        "a = %p\n",
        (void *)cache->a
    );

    printf(
        "z = %p\n",
        (void *)cache->z
    );

    printf("\nEliminando cache sin A ni Z...\n");

    eliminarCacheLayer(&cache);

    printf(
        "cache después de eliminar: %p\n",
        (void *)cache
    );

    if (cache != NULL)
    {
        printf(
            "[ERROR] eliminarCacheLayer() no dejó el puntero a NULL.\n"
        );

        correcto = 0;
    }
    else
    {
        printf(
            "[OK] Cache vacío eliminado correctamente.\n"
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
    printf("                                TEST CACHE\n");
    printf("===============================================================================\n");
    printf("\n");

    printf("Funciones evaluadas:\n");
    printf("  - inicializarCache()\n");
    printf("  - guardarCacheLayerA()\n");
    printf("  - guardarCacheLayerZ()\n");
    printf("  - eliminarCacheLayer()\n");
    printf("\n");

    printf(
        "NOTA: guardarCacheLayer() NO se utiliza porque está declarada\n"
        "      pero no implementada en el proyecto.\n"
    );

    printf("\n");
    printf("Tolerancia absoluta: %.2e\n", TOL_ABS);
    printf("Tolerancia relativa: %.2e\n", TOL_REL);

    separador();

    test_inicializarCache();

    separador();

    test_guardarA();
    test_guardarZ();

    separador();

    test_guardarA_y_Z();

    separador();

    test_sustituirA();
    test_sustituirZ();

    separador();

    test_independencia_A_Z();

    separador();

    test_copia_independiente_A();
    test_copia_independiente_Z();

    separador();

    test_dimensiones_variadas();

    separador();

    test_valores_variados();

    separador();

    test_eliminarCache();
    test_eliminarCacheVacio();

    separador();

    printf("\n");
    printf("===============================================================================\n");
    printf("                                  RESUMEN\n");
    printf("===============================================================================\n");

    printf("Tests ejecutados : %d\n", tests_totales);
    printf("Tests pasados    : %d\n", tests_pasados);
    printf("Tests fallados   : %d\n", tests_fallados);

    printf("\n");

    if (tests_fallados == 0)
    {
        printf("RESULTADO GLOBAL: PASS\n");
        printf(
            "Todos los tests de cache han pasado correctamente.\n"
        );
    }
    else
    {
        printf("RESULTADO GLOBAL: FAIL\n");
        printf(
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