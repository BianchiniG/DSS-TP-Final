#!/usr/bin/python

import os
import sys
import tarfile
from zipfile import ZipFile
from Preprocesamiento import Preprocesamiento
from modelos.RandomForest import RandomForest
from modelos.SVM import SVM

DATABASES_BASEPATH = '/app/datos/datasets/'
GOOGLESET_COMPRESSED_FILE = '/app/datos/datasets/faces-googleset.zip'
FACESDB_COMPRESSED_FILE = '/app/datos/datasets/faces-db.tar.xz'
FER_FILE = '/app/datos/datasets/fer2013.csv'
MODELOS = ['rf', 'svm', 'cnn']


def usage():
    print("""Modo de uso: 
    python command.py <args>
        * Inicializar:
            - args: init
        * Entrenar modelo:
            - args: <rf|svm|cnn> fit
        * Ejecutar predicción:
            - args: <rf|svm|cnn> predict <url_imagen>
        * Ayuda:
            - args: h
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


def exec_command(argumentos):
    print("Executing...")
    if len(argumentos) == 1:
        wrong_call()
    elif argumentos[1] == 'init':
        init_app_databases()
    elif argumentos[1] == 'h':
        usage()
        exit(0)
    elif argumentos[1] in MODELOS:
        modelo = get_modelo(argumentos[1])
        if argumentos[2] == 'fit':
            modelo.fit()
        elif argumentos[2] == 'predict':
            if len(argumentos) < 4:
                usage()
                exit(1)
            else:
                modelo.predict(argumentos[3])
    else:
        wrong_call()


def init_app_databases():
    print("Initializing...")
    p = Preprocesamiento()
    print("> Generando archivos de bases de datos")
    generate_db_files()
    print(">> Procesando fer database")
    p.export_fer_dataset()
    print(">> Generando...")
    p.preprocess_databases()
    init_cleanup()


def generate_db_files():
    extract_files(GOOGLESET_COMPRESSED_FILE, DATABASES_BASEPATH+'faces-googleset', 'zip')
    extract_files(FACESDB_COMPRESSED_FILE, DATABASES_BASEPATH+'faces-db', 'tar')


def extract_files(filename, target_extraction_folder, type):
    try:
        if type == 'zip':
            zf = ZipFile(filename)
            zf.extractall(path=target_extraction_folder)
            zf.close()
        elif type == 'tar':
            tar = tarfile.open(filename)
            tar.extractall(target_extraction_folder)
            tar.close()
        else:
            error("El tipo del archivo es incorrecto")
            return
    except FileNotFoundError:
        error("El archivo "+filename+" no existe")


def init_cleanup():
    for filename in [GOOGLESET_COMPRESSED_FILE, FACESDB_COMPRESSED_FILE, FER_FILE]:
        if os.path.exists(filename):
            os.remove(filename)


def wrong_call():
    usage()
    exit(1)


def error(mensaje):
    print(mensaje)
    exit(1)


exec_command(sys.argv)
