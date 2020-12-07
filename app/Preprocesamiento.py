import cv2
import os
import csv
import imageio
import numpy as np
import pandas as pd
from pathlib import Path
from modelos.utiles import DB_BASEPATH, IMG_COLS, IMG_ROWS, EMOCIONES, FER_DATASET_MAP, EMOCIONES

FACESDB_FOLDER = 'datasets/faces-db'
FACESGOOGLESET_FOLDER = 'datasets/faces-googleset'
FACESDB_OUT_FILENAME = 'faces-db.csv'
FACESGOOGLESET_OUT_FILENAME = 'faces-googleset-db.csv'
FER_ORIGINAL_ROUTE = '/app/datos/datasets/fer2013.csv'
FER_FOLDER = 'datasets/fer'
FER_OUT_FILENAME = 'fer.csv'
EMOCIONES_MAP = {
    'anger': 'anger',
    'fear': 'fear',
    'joy': 'happy',
    'happy': 'happy',
    'kiss': 'kiss',
    'neutral': 'neutral',
    'sadness': 'sadness',
    'disgust': 'disgust',
    'surprise': 'surprise'
}


class Preprocesamiento:
    def __init__(self):
        self.facesdb_folder = Path(DB_BASEPATH+'/'+FACESDB_FOLDER)
        self.facesgoogleset_folder = Path(DB_BASEPATH+'/'+FACESGOOGLESET_FOLDER)
        self.fer_folder = Path(DB_BASEPATH+'/'+FER_FOLDER)

    @staticmethod
    def preprocess_image(imagen):
        grayed_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_detected = face_cascade.detectMultiScale(grayed_img, scaleFactor=1.5, minNeighbors=5)
        if len(faces_detected):
            (x, y, w, h) = faces_detected[0]
            cv2.rectangle(grayed_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            img = cv2.normalize(grayed_img, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.resize(img[y:y+h, x:x+w], (IMG_COLS, IMG_ROWS))
        else:
            raise Exception()

    def preprocess_databases(self):
        if self.__databases_not_created():
            self.__create_dbs()

        databases = [DB_BASEPATH+FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME,
                     DB_BASEPATH+FACESGOOGLESET_FOLDER+'/'+FACESGOOGLESET_OUT_FILENAME]
        for database in databases:
            df = pd.read_csv(database)

            for index, row in df.iterrows():
                try:
                    imagen = cv2.imread(os.path.join(DB_BASEPATH, row.imagen))
                    cv2.imwrite(os.path.join(DB_BASEPATH, str(row.imagen)), self.preprocess_image(imagen))
                except Exception:
                    print("No se detecto una cara en la foto: "+os.path.join(DB_BASEPATH, row.imagen))
                    os.remove(os.path.join(DB_BASEPATH, row.imagen))
                    df = df.drop(index)
            os.remove(database)
            df.to_csv(database)



    def __create_dbs(self):
        dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
        self.__generate_db(self.facesdb_folder, dataframe)
        dataframe.to_csv(DB_BASEPATH+'/'+FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME)
        dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
        self.__generate_db(self.facesgoogleset_folder, dataframe)
        dataframe.to_csv(DB_BASEPATH+'/'+FACESGOOGLESET_FOLDER+'/'+FACESGOOGLESET_OUT_FILENAME)
        dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
        self.__generate_db(self.fer_folder, dataframe)
        dataframe.to_csv(DB_BASEPATH+'/'+FER_FOLDER+'/'+FER_OUT_FILENAME)

    def __generate_db(self, folder, dataframe):
        for file in os.scandir(folder):
            if os.path.isdir(file):
                self.__generate_db(file, dataframe)
            else:
                _, extension = os.path.splitext(file)
                if not extension == '.csv':
                    index = dataframe.count().imagen + 1
                    emocion = [EMOCIONES_MAP[e] for e in EMOCIONES_MAP.keys() if e in file.path][0]
                    datos = {'index': index, 'imagen': file.path, 'clase': emocion}
                    dataframe.loc[index] = datos

    def export_fer_dataset(self, ruta=FER_ORIGINAL_ROUTE):
        with open(ruta, 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            headers = next(datareader)

            id = 1
            for row in datareader:
                emotion = EMOCIONES[FER_DATASET_MAP[int(row[0])]]
                pixels = list(map(int, row[1].split()))
                print("Exportando imagen "+str(id)+" - emocion "+str(emotion))

                pixels_array = np.asarray(pixels)

                image = pixels_array.reshape(IMG_ROWS, IMG_COLS)
                image = image.astype(np.uint8)
                image_folder = os.path.join(DB_BASEPATH+'/'+FER_FOLDER, emotion)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_file = os.path.join(image_folder, str(id) + '.jpg')
                imageio.imwrite(image_file, image)
                id += 1

    @staticmethod
    def __databases_not_created():
        return not os.path.exists(DB_BASEPATH+'/'+FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME) or not \
            os.path.exists(DB_BASEPATH+'/'+FACESGOOGLESET_FOLDER + '/' + FACESGOOGLESET_OUT_FILENAME) or not \
            os.path.exists(DB_BASEPATH+'/'+FER_FOLDER+'/'+FER_OUT_FILENAME)
