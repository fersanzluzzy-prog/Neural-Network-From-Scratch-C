#include <stdio.h>
#include <stdlib.h>

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