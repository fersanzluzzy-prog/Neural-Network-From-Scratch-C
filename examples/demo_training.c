#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "network.h"
#include "layer.h"
#include "matrix.h"
#include "loss.h"

#define PI 3.14159265358979323846f

#define XOR_EPOCHS 400
#define XOR_LR 0.5f

#define CIRCLE_EPOCHS 250
#define CIRCLE_LR 0.08f
#define CIRCLE_SAMPLES 120

#define GRID_SIZE 50
#define SNAPSHOT_EVERY 10

struct sample {
    float x;
    float y;
    float target;
};

static void fail(const char *message)
{
    fprintf(stderr, "ERROR: %s\n", message);
    exit(EXIT_FAILURE);
}

static void set_value(struct matrix *m, int row, int col, float value)
{
    m->datos[row * m->col + col] = value;
}

static void clear_caches(struct network *red)
{
    for (int i = 0; i < red->numeroCapas; i++) {
        if (red->cache[i] != NULL) {
            eliminarCacheLayer(&red->cache[i]);
        }
    }
}

/*
 * Nuestro backprop actual reserva un nuevo gradientesLayer en cada
 * llamada. Para que este ejemplo visual no acumule gradientes viejos,
 * los liberamos justo después de usar descensoDeGradiente().
 */
static void clear_gradients(struct network *red)
{
    for (int i = 0; i < red->numeroCapas; i++) {
        struct gradientesLayer *g = red->capas[i]->gradientes;

        if (g == NULL) {
            continue;
        }

        eliminarMatriz(&g->dL_dw);
        eliminarMatriz(&g->dL_db);
        eliminarMatriz(&g->dL_dx);

        free(g);
        red->capas[i]->gradientes = NULL;
    }
}

/*
 * W = nSalidas x nEntradas
 * b = nSalidas x 1
 */
static void create_random_layer(
    int nEntradas,
    int nSalidas,
    activacion act,
    struct layer **resultado)
{
    struct matrix *w = NULL;
    struct matrix *b = NULL;

    crearMatriz(&w, nSalidas, nEntradas);
    crearMatriz(&b, nSalidas, 1);

    float limite = sqrtf(6.0f / (float)(nEntradas + nSalidas));

    inicializarRandom(w, -limite, limite);

    for (int i = 0; i < nSalidas; i++) {
        b->datos[i] = 0.0f;
    }

    if (crearLayer(w, b, act, resultado) != 0) {
        eliminarMatriz(&w);
        eliminarMatriz(&b);
        fail("crearLayer ha fallado.");
    }

    eliminarMatriz(&w);
    eliminarMatriz(&b);
}

static struct network *create_xor_network(void)
{
    struct network *red = NULL;
    struct layer *hidden = NULL;
    struct layer *output = NULL;

    crearRed(&red);

    create_random_layer(2, 4, TANH, &hidden);
    create_random_layer(4, 1, SIGMOID, &output);

    if (añadirCapa(hidden, red) != 0 ||
        añadirCapa(output, red) != 0) {
        fail("No se ha podido construir la red XOR.");
    }

    return red;
}

static struct network *create_circle_network(void)
{
    struct network *red = NULL;
    struct layer *hidden = NULL;
    struct layer *output = NULL;

    crearRed(&red);

    create_random_layer(2, 8, TANH, &hidden);
    create_random_layer(8, 1, SIGMOID, &output);

    if (añadirCapa(hidden, red) != 0 ||
        añadirCapa(output, red) != 0) {
        fail("No se ha podido construir la red de círculos.");
    }

    return red;
}

static void create_xor_dataset(struct sample data[4])
{
    data[0] = (struct sample){0.0f, 0.0f, 0.0f};
    data[1] = (struct sample){0.0f, 1.0f, 1.0f};
    data[2] = (struct sample){1.0f, 0.0f, 1.0f};
    data[3] = (struct sample){1.0f, 1.0f, 0.0f};
}

static float random_float(float min, float max)
{
    return min + (max - min) *
        ((float)rand() / (float)RAND_MAX);
}

static void create_circle_dataset(struct sample data[CIRCLE_SAMPLES])
{
    const int half = CIRCLE_SAMPLES / 2;

    for (int i = 0; i < half; i++) {
        float angle = random_float(0.0f, 2.0f * PI);
        float radius = random_float(0.05f, 0.60f);

        data[i].x = radius * cosf(angle);
        data[i].y = radius * sinf(angle);
        data[i].target = 0.0f;
    }

    for (int i = half; i < CIRCLE_SAMPLES; i++) {
        float angle = random_float(0.0f, 2.0f * PI);
        float radius = random_float(0.90f, 1.05f);

        data[i].x = radius * cosf(angle);
        data[i].y = radius * sinf(angle);
        data[i].target = 1.0f;
    }
}

static void write_dataset(
    const char *path,
    const struct sample *data,
    int n)
{
    FILE *file = fopen(path, "w");

    if (file == NULL) {
        fail("No se ha podido crear el CSV del dataset.");
    }

    fprintf(file, "x,y,label\n");

    for (int i = 0; i < n; i++) {
        fprintf(file, "%.9f,%.9f,%.0f\n",
                data[i].x,
                data[i].y,
                data[i].target);
    }

    fclose(file);
}

static float calculate_dataset_loss(
    struct network *red,
    const struct sample *data,
    int n)
{
    float total = 0.0f;

    for (int i = 0; i < n; i++) {
        struct matrix *input = NULL;
        struct matrix *expected = NULL;
        struct matrix *output = NULL;

        crearMatriz(&input, 2, 1);
        crearMatriz(&expected, 1, 1);

        set_value(input, 0, 0, data[i].x);
        set_value(input, 1, 0, data[i].y);
        set_value(expected, 0, 0, data[i].target);

        output = forwardRed(red, input);

        if (output == NULL) {
            fail("forwardRed ha fallado calculando el loss.");
        }

        total += mse(*output, *expected);

        eliminarMatriz(&output);
        eliminarMatriz(&input);
        eliminarMatriz(&expected);

        clear_caches(red);
    }

    return total / (float)n;
}

static void train_one_epoch(
    struct network *red,
    const struct sample *data,
    int n,
    float learning_rate)
{
    for (int i = 0; i < n; i++) {
        struct matrix *input = NULL;
        struct matrix *expected = NULL;
        struct matrix *output = NULL;

        crearMatriz(&input, 2, 1);
        crearMatriz(&expected, 1, 1);

        set_value(input, 0, 0, data[i].x);
        set_value(input, 1, 0, data[i].y);
        set_value(expected, 0, 0, data[i].target);

        output = forwardRed(red, input);

        if (output == NULL) {
            fail("forwardRed ha fallado durante el entrenamiento.");
        }

        if (backpropRed(red, *output, *expected) != 0) {
            fail("backpropRed ha devuelto error.");
        }

        if (descensoDeGradiente(red, learning_rate) != 0) {
            fail("descensoDeGradiente ha devuelto error.");
        }

        eliminarMatriz(&output);
        eliminarMatriz(&input);
        eliminarMatriz(&expected);

        clear_gradients(red);
        clear_caches(red);
    }
}

static void write_snapshot(
    FILE *file,
    struct network *red,
    int epoch,
    float loss)
{
    /*
     * El separador "#" permite guardar metadatos sin complicar
     * el formato del resto del CSV.
     */
    fprintf(file, "# EPOCH %d LOSS %.9f\n", epoch, loss);

    const float min = -1.1f;
    const float max = 1.1f;
    const float step = (max - min) / (float)(GRID_SIZE - 1);

    for (int row = 0; row < GRID_SIZE; row++) {
        float y = min + row * step;

        for (int col = 0; col < GRID_SIZE; col++) {
            float x = min + col * step;

            struct matrix *input = NULL;
            struct matrix *output = NULL;

            crearMatriz(&input, 2, 1);
            set_value(input, 0, 0, x);
            set_value(input, 1, 0, y);

            output = forwardRed(red, input);

            if (output == NULL) {
                fail("forwardRed ha fallado creando la frontera.");
            }

            fprintf(file, "%.9f,%.9f,%.9f,%d\n",
                    x, y, output->datos[0], epoch);

            eliminarMatriz(&output);
            eliminarMatriz(&input);
            clear_caches(red);
        }
    }

    fflush(file);
}

static void run_demo(
    const char *name,
    struct network *red,
    const struct sample *data,
    int n,
    int epochs,
    float learning_rate)
{
    char data_path[256];
    char training_path[256];

    snprintf(data_path, sizeof(data_path),
             "examples/output/%s_data.csv", name);

    snprintf(training_path, sizeof(training_path),
             "examples/output/%s_training.csv", name);

    write_dataset(data_path, data, n);

    FILE *file = fopen(training_path, "w");

    if (file == NULL) {
        fail("No se ha podido crear el CSV del entrenamiento.");
    }

    fprintf(file, "x,y,prediction,epoch\n");

    float loss = calculate_dataset_loss(red, data, n);
    write_snapshot(file, red, 0, loss);

    printf("%s | epoch %d/%d | loss %.6f\n",
           name, 0, epochs, loss);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        train_one_epoch(red, data, n, learning_rate);
        loss = calculate_dataset_loss(red, data, n);

        if (epoch % SNAPSHOT_EVERY == 0 || epoch == epochs) {
            write_snapshot(file, red, epoch, loss);

            printf("%s | epoch %d/%d | loss %.6f\n",
                   name, epoch, epochs, loss);
        }
    }

    fclose(file);
}

int main(void)
{
    /*
     * Semilla fija para que el experimento sea reproducible.
     */
    srand(42);

    /*
     * La carpeta examples/output debe existir.
     * En Windows se puede crear manualmente una vez.
     */
    struct sample xor_data[4];
    struct sample circle_data[CIRCLE_SAMPLES];

    create_xor_dataset(xor_data);
    create_circle_dataset(circle_data);

    printf("============================================================\n");
    printf(" DEMO DE ENTRENAMIENTO: XOR + CIRCULOS CONCENTRICOS\n");
    printf("============================================================\n\n");

    printf("Entrenando XOR...\n");
    struct network *xor_red = create_xor_network();

    run_demo(
        "xor",
        xor_red,
        xor_data,
        4,
        XOR_EPOCHS,
        XOR_LR
    );

    eliminarRed(&xor_red);

    printf("\nEntrenando circulos concentricos...\n");
    struct network *circle_red = create_circle_network();

    run_demo(
        "circles",
        circle_red,
        circle_data,
        CIRCLE_SAMPLES,
        CIRCLE_EPOCHS,
        CIRCLE_LR
    );

    eliminarRed(&circle_red);

    printf("\nDatos guardados en examples/output/\n");

    return 0;
}
