cmake_minimum_required(VERSION 3.20)

project(
    NeuralNetwork
    VERSION 1.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Alertas
add_compile_options(-Wall -Wextra -Wpedantic)

# Librería
add_library(NeuralNetwork
    src/matrix.c
    src/layer.c
    src/activation.c

)

target_include_directories(NeuralNetwork PUBLIC include)

# =========================
# Tests
# =========================

add_executable(test_matrix tests/test_matrix.c)
target_link_libraries(test_matrix NeuralNetwork)

add_executable(test_layer tests/test_layer.c)
target_link_libraries(test_layer NeuralNetwork)

add_executable(test_matrix_mult tests/test_matrix_mult.c)
target_link_libraries(test_matrix_mult NeuralNetwork)

add_executable(test_activation tests/test_activation.c)
target_link_libraries(test_activation NeuralNetwork)