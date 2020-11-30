import abc
import numpy as np
import pandas as pd
from cv2 import imread
from utiles import get_label_by_emotion, DB_BASEPATH, FACESDB_ROUTE, FACESGOOGLESET_ROUTE, EMOCIONES


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self):
        """
        Acá entrenamos el modelo con los datos preprocesados (Se implementa en cada clase concreta).
        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, image):
        """
        Acá, en base al modelo ya entrenado, predecimos qué emocion contiene la cara de la imagen que nos mandan (Se
        implementa en cada clase concreta).
        :param image:
        :return:
        """
        pass

    @staticmethod
    def get_images_for_training():
        images_with_emotions = {}

        for label, emotion in EMOCIONES.items():
            images_with_emotions[label] = []

        faces_db = pd.read_csv(FACESDB_ROUTE)
        for index, row in faces_db.iterrows():
            images_with_emotions[get_label_by_emotion(row.clase)].append(imread(DB_BASEPATH+row.imagen))
        faces_google_set_db = pd.read_csv(FACESGOOGLESET_ROUTE)
        for index, row in faces_google_set_db.iterrows():
            images_with_emotions[get_label_by_emotion(row.clase)].append(imread(DB_BASEPATH+row.imagen))

        return images_with_emotions
