import cv2
import os
from pathlib import Path
import pandas as pd
from modelos.utiles import DB_BASEPATH, IMG_COLS, IMG_ROWS

FACESDB_FOLDER = 'datasets/faces-db'
FACESGOOGLESET_FOLDER = 'datasets/faces-googleset'
FACESDB_OUT_FILENAME = 'faces-db.csv'
FACESGOOGLESET_OUT_FILENAME = 'faces-googleset-db.csv'
EMOCIONES = {
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

    @staticmethod
    def preprocess_image(imagen):
        grayed_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_detected = face_cascade.detectMultiScale(grayed_img, scaleFactor=1.5, minNeighbors=5)
        if faces_detected:
            (x, y, w, h) = faces_detected[0]
            cv2.rectangle(grayed_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            img = cv2.normalize(grayed_img, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.resize(img[y:y+h, x:x+w], (IMG_COLS, IMG_ROWS))
        else:
            raise Exception()

    def preprocess_databases(self):
        if self.__databases_not_created():
            self.__create_dbs()

        databases = [DB_BASEPATH+'/'+FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME,
                     DB_BASEPATH+'/'+FACESGOOGLESET_FOLDER+'/'+FACESGOOGLESET_OUT_FILENAME]
        for database in databases:
            df = pd.read_csv(database)

            for index, row in df.iterrows():
                try:
                    imagen = cv2.imread(os.path.join(DB_BASEPATH, row.imagen))
                    cv2.imwrite(os.path.join(DB_BASEPATH, str(row.imagen)), self.preprocess_image(imagen))
                except Exception:
                    print("No se detecto una cara en la foto: "+os.path.join(DB_BASEPATH, row.imagen))

    def __create_dbs(self):
        dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
        self.__generate_db(self.facesdb_folder, dataframe)
        dataframe.to_csv(DB_BASEPATH+'/'+FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME)
        dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
        self.__generate_db(self.facesgoogleset_folder, dataframe)
        dataframe.to_csv(DB_BASEPATH+'/'+FACESGOOGLESET_FOLDER+'/'+FACESGOOGLESET_OUT_FILENAME)

    def __generate_db(self, folder, dataframe):
        for file in os.scandir(folder):
            if os.path.isdir(file):
                self.__generate_db(file, dataframe)
            else:
                _, extension = os.path.splitext(file)
                if not extension == '.csv':
                    index = dataframe.count().imagen + 1
                    print(file.path)
                    emocion = [EMOCIONES[e] for e in EMOCIONES.keys() if e in file.path][0]
                    datos = {'index': index, 'imagen': file.path, 'clase': emocion}
                    dataframe.loc[index] = datos

    @staticmethod
    def __databases_not_created():
        return not os.path.exists(DB_BASEPATH+'/'+FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME) or not \
            os.path.exists(DB_BASEPATH+'/'+FACESGOOGLESET_FOLDER + '/' + FACESGOOGLESET_OUT_FILENAME)
