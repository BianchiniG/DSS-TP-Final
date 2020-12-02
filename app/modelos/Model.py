import abc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import imread, cvtColor, COLOR_RGB2GRAY, resize
from utiles import get_label_by_emotion, DB_BASEPATH, FACESDB_ROUTE, FACESGOOGLESET_ROUTE, EMOCIONES

IMG_ROWS = 48
IMG_COLS = 48


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

    def get_images_and_labels_for_training(self):
        images = []
        labels = []

        faces_db = pd.read_csv(FACESDB_ROUTE)
        for index, row in faces_db.iterrows():
            images.append(DB_BASEPATH+row.imagen)
            labels.append(get_label_by_emotion(row.clase))
        faces_google_set_db = pd.read_csv(FACESGOOGLESET_ROUTE)
        for index, row in faces_google_set_db.iterrows():
            images.append(DB_BASEPATH+row.imagen)
            labels.append(get_label_by_emotion(row.clase))

        return images, labels

    @staticmethod
    def plot_confusion_matrix(cm, classes=EMOCIONES.values(), normalize=True, title='Confusion matrix', cmap=plt.cm.Greens, archivo='plt.png'):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized ' + title

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(archivo)

    def plot_classification_report(self):
        # classification_report(y_test, y_pred)
        pass

    def __get_image_data(self, image):
        return resize(cvtColor(imread(image), COLOR_RGB2GRAY), (IMG_ROWS, IMG_COLS))
