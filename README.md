# Neural Network From Scratch in C

An educational project to build a fully connected neural network framework entirely from scratch in C, understanding every algorithm instead of relying on existing libraries.

The goal of this project is not only to build a working neural network, but also to understand every mathematical concept, algorithm, and data structure behind modern deep learning frameworks.

---

## Current Features

### Matrix Module

Implemented operations:

- Matrix creation and destruction
- Matrix initialization from an array
- Constant matrix initialization
- Random matrix initialization
- Element access and modification
- Matrix transpose
- Matrix addition
- Matrix subtraction
- Scalar multiplication
- Matrix multiplication

---

### Layer Module

Implemented features:

- Dense (fully connected) layer
- Forward propagation
- Automatic dimension checking
- Configurable activation function

---

### Activation Module

Implemented activation functions:

- ReLU
- Sigmoid
- Tanh

---

### Network Module

Implemented features:

- Network creation
- Dynamic layer addition
- Layer compatibility validation
- Network destruction

---

## Project Goals

- Build a fully connected neural network from scratch.
- Implement forward propagation.
- Implement loss functions.
- Implement backpropagation.
- Implement gradient descent optimization.
- Explore different weight initialization strategies.
- Optimize performance while keeping the implementation educational.

---

## Technologies

- Language: C11
- Build system: CMake
- Compiler: GCC

---

## Project Structure

```
.
├── include/
│   ├── activation.h
│   ├── layer.h
│   ├── matrix.h
│   └── network.h
│
├── src/
│   ├── activation.c
│   ├── layer.c
│   ├── matrix.c
│   └── network.c
│
├── tests/
│   ├── test_activation.c
│   ├── test_layer.c
│   ├── test_matrix.c
│   ├── test_matrix_mult.c
│   └── test_network.c
│
├── build/
├── CMakeLists.txt
└── README.md
```

---

## Current Status

### ✅ Completed

- Matrix module
- Layer module
- Activation module
- Network module
- Unit tests for every module

### 🚧 Next Milestones

- Complete network forward propagation
- Loss functions
- Backpropagation
- Optimizers
- Training loop