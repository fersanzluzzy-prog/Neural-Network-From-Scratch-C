# Neural Network From Scratch in C

Implementación de una red neuronal desde cero en **C**, sin utilizar frameworks de deep learning. El objetivo del proyecto ha sido entender y construir manualmente las piezas fundamentales que intervienen en el funcionamiento de una red neuronal, desde las operaciones matriciales hasta el entrenamiento mediante backpropagation y descenso de gradiente.

## Uso

### 1. Compilar el proyecto

El proyecto utiliza CMake y MinGW para la compilación. Desde la carpeta raíz del repositorio:

```bash
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
```

> **Estado:** primera versión completada.  
> Esta versión se considera una base funcional sobre la que seguiré desarrollando y experimentando con la librería.

---

## ¿Qué contiene el proyecto?

La librería está dividida en varios módulos, cada uno encargado de una parte concreta de la red.

```text
include/
├── matrix.h
├── activation.h
├── loss.h
├── layer.h
├── network.h
└── training.h

src/
├── matrix.c
├── activation.c
├── loss.c
├── layer.c
├── network.c
└── training.c
```

### Matrices

Se ha implementado una estructura propia:

```c
struct matrix {
    int fil;
    int col;
    float *datos;
};
```

y operaciones básicas como:

- creación y eliminación de matrices;
- acceso y modificación de posiciones;
- inicialización;
- copia profunda;
- valores aleatorios;
- transposición;
- suma y resta;
- multiplicación por escalar;
- multiplicación matricial;
- multiplicación elemento a elemento.

La librería utiliza estas operaciones como base del resto de componentes de la red.

### Funciones de activación

Actualmente se incluyen:

- ReLU
- Sigmoid
- Tanh

y sus correspondientes derivadas para el cálculo de gradientes.

### Capas

Cada capa contiene sus pesos, bias, número de entradas y salidas, función de activación y gradientes.

Además, se utiliza una caché de las activaciones `a` y valores `z` obtenidos durante el forward para poder realizar posteriormente el backpropagation.

### Función de pérdida

Actualmente se utiliza **Mean Squared Error (MSE)** junto con su derivada.

### Forward propagation

La red puede encadenar varias capas y realizar:

z = Wx + b

seguido de la función de activación correspondiente.

### Backpropagation

Se ha implementado el cálculo manual de los gradientes de:

- `dL/dW`
- `dL/db`
- `dL/dx`

para propagar el error hacia las capas anteriores.

Los gradientes se comprueban mediante casos matemáticos conocidos y comprobaciones numéricas.

### Descenso de gradiente

La actualización de los parámetros sigue la regla:

W = W - ηdW   o    B = B - ηdB

donde `η` es el learning rate.

### Entrenamiento

Actualmente el entrenamiento se realiza **sin batches**: cada muestra se procesa de forma individual.

El flujo de entrenamiento es:

```text
entrada
   ↓
forward
   ↓
loss
   ↓
backpropagation
   ↓
gradientes
   ↓
descenso de gradiente
   ↓
siguiente muestra
```

Este proceso se repite durante el número de épocas indicado.

---

## Tests

El proyecto contiene una batería de tests independientes para comprobar las diferentes partes de la librería:

```text
tests/
├── test_matrix.c
├── test_layer.c
├── test_activation.c
├── test_network.c
├── test_cache.c
├── test_loss.c
├── test_descensoGradiente.c
└── test_training.c
```

Los tests comprueban tanto casos normales como casos límite y errores de dimensiones.

Especialmente, se ha utilizado **gradient checking numérico** para comparar los gradientes calculados por backpropagation con aproximaciones numéricas de las derivadas. Esto permite detectar errores sutiles en la propagación del gradiente.

---

## Ejemplo visual de entrenamiento

El proyecto incluye ejemplos sencillos para visualizar cómo aprende la red.

### XOR

Se entrena una red sobre la puerta lógica XOR:

```text
Entrada 1 | Entrada 2 | Salida
-----------+-----------+-------
    0      |     0     |   0
    0      |     1     |   1
    1      |     0     |   1
    1      |     1     |   0
```

Este ejemplo muestra cómo una red con una capa oculta aprende una frontera de decisión no lineal.

### Círculos concéntricos

También se utiliza un dataset generado artificialmente formado por dos clases distribuidas en círculos concéntricos.

Este problema es más complejo visualmente y sirve para observar cómo una red neuronal puede aprender una frontera de decisión cerrada y no lineal.

Los ejemplos de entrenamiento generan datos intermedios que posteriormente pueden representarse para observar la evolución de la frontera de decisión.

---

## Compilación

El proyecto utiliza **CMake**.

Con MinGW:

```powershell
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
```

Los tests pueden compilarse como targets independientes.

Por ejemplo:

```powershell
cmake --build build --target test_network
```

o:

```powershell
cmake --build build --target test_training
```

---

## Tecnologías

- C11
- CMake
- MinGW / GCC
- Python para los ejemplos de visualización
- Matplotlib para representar el entrenamiento

No se utilizan frameworks de redes neuronales para la implementación de la red.

---

## Objetivos del proyecto

El propósito principal no ha sido crear una librería competitiva con frameworks como PyTorch o TensorFlow, sino construir una implementación propia para entender en profundidad qué sucede internamente en una red neuronal.

La implementación parte de conceptos básicos como:

```text
Matrices
   ↓
Capas
   ↓
Forward
   ↓
Loss
   ↓
Backpropagation
   ↓
Gradientes
   ↓
Descenso de gradiente
   ↓
Entrenamiento
```

Esto permite utilizar la librería como base para experimentar con diferentes algoritmos y optimizaciones sin depender de una implementación de alto nivel.

---

## Próximos pasos

Aunque esta primera versión está terminada, **el proyecto no pretende quedarse en este estado**.

La librería está pensada como una base sobre la que seguir experimentando. Algunas de las líneas de desarrollo que quiero explorar son:

### Optimización

Estudiar y comparar diferentes formas de acelerar la implementación:

- optimización de las operaciones matriciales;
- mejora de la gestión de memoria;
- reducción de asignaciones dinámicas;
- vectorización y SIMD;
- paralelización;
- comparación de diferentes precisiones numéricas;
- mini-batches;
- otras estrategias de optimización del entrenamiento.

### Convoluciones

Uno de los siguientes pasos que quiero investigar es ampliar la librería para soportar **redes convolucionales (CNN)**.

Esto implicaría incorporar nuevas operaciones y estructuras para trabajar con datos multidimensionales y capas convolucionales.

### Nuevas arquitecturas y algoritmos

La arquitectura actual también puede servir como base para estudiar:

- nuevos tipos de capas;
- nuevos optimizadores;
- nuevas funciones de pérdida;
- mejores métodos de inicialización;
- regularización;
- arquitecturas más complejas.


---

## Filosofía del proyecto

Este proyecto sigue una idea muy sencilla:

> **Entender cómo funciona una red neuronal implementándola desde cero.**

En lugar de ocultar las operaciones detrás de un framework, cada parte importante de la red se implementa explícitamente para poder estudiarla, probarla y modificarla.

La intención es seguir utilizando esta librería como laboratorio para experimentar con técnicas de aprendizaje automático y optimización a bajo nivel.

También cabe mencionar la utilización de herramientas de inteligencia artificial **únicamente** para la creación de tests/ y examples/, al igual que para aprender. Al tener un fin puramente didáctico no se ha utilizado para generar ningún aspecto de la lógica de la red, aunque esto haya supuesto dolores de cabeza.

---

## Estado actual

### Implementado

- [x] Sistema de matrices propio
- [x] Operaciones matriciales
- [x] ReLU
- [x] Sigmoid
- [x] Tanh
- [x] Capas densas
- [x] Forward propagation
- [x] Caché de activaciones
- [x] MSE
- [x] Backpropagation
- [x] Gradientes de pesos y bias
- [x] Descenso de gradiente
- [x] Entrenamiento por muestras
- [x] Tests de los principales módulos
- [x] Gradient checking
- [x] Ejemplo XOR
- [x] Ejemplo de círculos concéntricos
- [x] Visualización del entrenamiento

### Pendiente / en investigación

- [ ] Mini-batches
- [ ] Optimizadores adicionales
- [ ] SIMD / vectorización
- [ ] Paralelización
- [ ] CUDA
- [ ] Redes convolucionales
- [ ] Nuevos tipos de capas
- [ ] Más datasets y experimentos
- [ ] Herramientas adicionales de análisis y benchmarking

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta `LICENSE` para más información.
