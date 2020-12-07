#!/usr/bin/python

import os
import sys
import shutil
import tarfile
from zipfile import ZipFile
from Preprocesamiento import Preprocesamiento
from modelos.RandomForest import RandomForest
from modelos.SVM import SVM
from modelos.utiles import DB_BASEPATH, \
    FACESGOOGLESET_ROUTE, \
    FACESDB_ROUTE, \
    FER_ROUTE, \
    FACESDB_FULL_PATH, \
    FACESGOOGLESETDB_FULL_PATH, \
    FERDB_FULL_PATH, \
    FACESDB_COMPRESSED_FILE, \
    FACESGOOGLESET_COMPRESSED_FILE, \
    FER_ORIGINAL_FILE

DATABASES_BASEPATH = '/app/datos/datasets/'
MODELOS = ['rf', 'svm', 'cnn']


def usage():
    print("""Modo de uso: 
    python command.py <args>
        * Inicializar bases de datos:
            - args: init
        * Inicializar bases de datos de landmarks (Cuidado! Es un proceso lento!):
            - args: init_landmarks
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
    elif argumentos[1] == 'init_landmarks':
        init_app_landmarks_databases()
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
    if not os.path.exists(DB_BASEPATH+FER_ROUTE) and os.path.exists(FER_ORIGINAL_FILE):
        os.mkdir(DB_BASEPATH+FER_ROUTE)
        p.export_fer_dataset()
    else:
        print(">>> El sataset de fer ya fue procesado")
    print(">> Generando...")
    p.preprocess_databases()
    init_cleanup()


def init_app_landmarks_databases():
    if os.path.exists(FACESDB_FULL_PATH) and os.path.exists(FACESGOOGLESETDB_FULL_PATH) and os.path.exists(FERDB_FULL_PATH):
        p = Preprocesamiento()
        p.generate_landmarks_dbs()


def generate_db_files():
    if os.path.exists(FACESDB_COMPRESSED_FILE) and not os.path.exists(DB_BASEPATH+FACESDB_ROUTE):
        extract_files(FACESDB_COMPRESSED_FILE, DATABASES_BASEPATH+'faces-db', 'tar')
        if os.path.exists(DB_BASEPATH+FACESDB_ROUTE+'kiss'):
            shutil.rmtree(DB_BASEPATH+FACESDB_ROUTE+'kiss')
    if os.path.exists(FACESGOOGLESET_COMPRESSED_FILE) and not os.path.exists(DB_BASEPATH+FACESGOOGLESET_ROUTE):
        extract_files(FACESGOOGLESET_COMPRESSED_FILE, DATABASES_BASEPATH+'faces-googleset', 'zip')


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
    for filename in [FACESGOOGLESET_COMPRESSED_FILE, FACESDB_COMPRESSED_FILE, FER_ORIGINAL_FILE]:
        if os.path.exists(filename):
            os.remove(filename)
    if os.path.exists(DB_BASEPATH+FACESDB_ROUTE+'kiss'):
        shutil.rmtree(DB_BASEPATH+FACESDB_ROUTE+'kiss')


def wrong_call():
    usage()
    exit(1)


def error(mensaje):
    print(mensaje)
    exit(1)


exec_command(sys.argv)
