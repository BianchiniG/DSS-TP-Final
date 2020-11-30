import numpy as np
import pandas as pd
from Model import Model
from utiles import EMOCIONES, RF_TRAINED_MODEL_FILE


class RandomForest(Model):
    def __init__(self):
        self.training_df = None

    def fit(self):
        images_with_emotions = self.get_images_for_training()
        self.__get_training_dataframe(images_with_emotions)
        print(self.training_df)
        # Guardar el modelo en RF_TRAINED_MODEL_FILE
        # Generar html con resultados en la carpeta templates

    def predict(self, image):
        # Levantar el modelo de RF_TRAINED_MODEL_FILE y realizar la predicción
        pass

    def __get_landmarks(self, image):
        pass

    def __get_training_dataframe(self, images_with_emotions):
        data = []
        for label, images in images_with_emotions.items():
            for image in images:
                data.append(np.concatenate((image.flatten(), [EMOCIONES[label]], )))  # TODO:
        self.training_df = pd.DataFrame(data)
        self.training_df['target'] = self.training_df[self.training_df.columns[-1]]
