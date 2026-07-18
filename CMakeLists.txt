cmake_minimum_required(VERSION 3.20)

project(
    NeuralNetwork
    VERSION 1.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_executable(
    NeuralNetwork
    src/main.c
)

target_include_directories(
    NeuralNetwork
    PRIVATE include
)