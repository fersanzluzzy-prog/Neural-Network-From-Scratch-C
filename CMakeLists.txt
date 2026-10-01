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
    src/network.c
    src/loss.c

)

target_include_directories(NeuralNetwork PUBLIC include)

# =========================
# Tests
# =========================

add_executable(test_matrix tests/test_matrix.c)
target_link_libraries(test_matrix NeuralNetwork)

add_executable(test_layer tests/test_layer.c)
target_link_libraries(test_layer NeuralNetwork)

add_executable(test_activation tests/test_activation.c)
target_link_libraries(test_activation NeuralNetwork)

add_executable(test_network tests/test_network.c)
target_link_libraries(test_network NeuralNetwork)

add_executable(test_cache tests/test_cache.c)
target_link_libraries(test_cache NeuralNetwork)

add_executable(test_loss tests/test_loss.c)
target_link_libraries(test_loss NeuralNetwork)