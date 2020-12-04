#!/usr/bin/python

import sys
from modelos.RandomForest import RandomForest
from modelos.SVM import SVM

def modo_de_uso():
    print("""Modo de uso: 
    python command.py <nombre_modelo> <fit>|<predict> <imagen>
        * El parámetro imagen es necesario solo para el predict
        * Los posibles modelos son:
            - rf (Random Forest)
            - svm (Máquina Vectorial)
            - cnn (Red Neuronal)
    """)


def get_modelo(modelo_elegido):
    if modelo_elegido == 'rf':
        return RandomForest()
    elif modelo_elegido == 'svm':
        return SVM()
    elif modelo_elegido == 'cnn':
        error("Modelo no implementado")
    else:
        error("El modelo elegido ("+modelo_elegido+") no existe")


def ejecutar_comando(modelo, opciones):
    if opciones[2] == 'fit':
        modelo.fit()
    elif opciones[2] == 'predict':
        if len(opciones) < 4:
            modo_de_uso()
            exit(1)
        else:
            modelo.predict(opciones[3])


def args_error(argumentos):
    if len(argumentos) == 1 or len(argumentos) < 3 or argumentos[1] == 'h':
        modo_de_uso()
        exit(1)
    else:
        return False


def error(mensaje):
    print(mensaje)
    exit(1)


if not args_error(sys.argv):
    ejecutar_comando(get_modelo(sys.argv[1]), sys.argv)
