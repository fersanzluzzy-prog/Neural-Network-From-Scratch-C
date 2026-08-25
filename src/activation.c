#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "activation.h"

//si sale algo mal -1, si no 1
int relu(struct matrix *entradas, struct matrix *salidas){
    if (entradas->fil == salidas->fil && entradas->col==salidas->col){

        for(int i=0; i < entradas->fil * entradas->col; i++){
            if(entradas->datos[i] > 0){salidas->datos[i] = entradas->datos[i];}
            else {salidas->datos[i] = 0;}
        }
        return 1;
    } 
    return -1;
}

int reluDerivada(struct matrix *entradas, struct matrix *salidas){
    if (entradas->fil == salidas->fil && entradas->col==salidas->col){

        for(int i=0; i < entradas->fil * entradas->col; i++){
            if(entradas->datos[i] > 0){salidas->datos[i] = 1;}
            else {salidas->datos[i] = 0;}
        }
        return 1;
    } 
    return -1;
}

int sigmoide(struct matrix *entradas, struct matrix *salidas){
    if (entradas->fil == salidas->fil && entradas->col==salidas->col){

        for(int i=0; i < entradas->fil * entradas->col; i++){
            salidas->datos[i]= 1/(1 + exp(-entradas->datos[i])); //funcion sigmoide
        }
        return 1;
    } 
    return -1;
}

//se deben pasar las entradas con la funcion sigmoide ya calculada
int sigmoideDerivada(struct matrix *entradasSigmoide, struct matrix *salidas){
    if (entradasSigmoide->fil == salidas->fil && entradasSigmoide->col==salidas->col){
        for(int i=0; i < entradasSigmoide->fil * entradasSigmoide->col; i++){
            salidas->datos[i]= entradasSigmoide->datos[i] * (1 - entradasSigmoide->datos[i]); //fsigmoide * (1 - fsigmoide)
        }
        return 1;
    } 
    return -1;
}

int taNh(struct matrix *entradas, struct matrix *salidas){
    if (entradas->fil == salidas->fil && entradas->col==salidas->col){
        for(int i=0; i < entradas->fil * entradas->col; i++){
            salidas->datos[i]= (exp(entradas->datos[i]) - exp(-entradas->datos[i])) / (exp(entradas->datos[i]) + exp(-entradas->datos[i])); //funcion tanh
        }
        return 1;
    } 
    return -1;
}

//hay que pasar las entradas con la función tanh ya calculada
int taNhDerivada(struct matrix *entradas, struct matrix *salidas){
    if (entradas->fil == salidas->fil && entradas->col==salidas->col){
        for(int i=0; i < entradas->fil * entradas->col; i++){
            salidas->datos[i]= 1 - pow(entradas->datos[i], 2);
        }
        return 1;
    } 
    return -1;
}