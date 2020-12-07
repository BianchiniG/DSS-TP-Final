import os
import abc
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import imread, cvtColor, COLOR_RGB2GRAY, resize
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from .utiles import get_label_by_emotion, \
    DB_BASEPATH, \
    FACESDB_FULL_PATH, \
    FACESGOOGLESETDB_FULL_PATH, \
    FERDB_FULL_PATH, \
    FACESDB_LANDMARKS_FULL_PATH, \
    FACESGOOGLESETDB_LANDMARKS_FULL_PATH, \
    FERDB_LANDMARKS_FULL_PATH, \
    EMOCIONES


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

        faces_db = pd.read_csv(FACESDB_FULL_PATH)
        for index, row in faces_db.iterrows():
            images.append(row.imagen)
            labels.append(get_label_by_emotion(row.clase))
        faces_google_set_db = pd.read_csv(FACESGOOGLESETDB_FULL_PATH)
        for index, row in faces_google_set_db.iterrows():
            images.append(row.imagen)
            labels.append(get_label_by_emotion(row.clase))
        fer_db = pd.read_csv(FERDB_FULL_PATH)
        for index, row in fer_db.iterrows():
            images.append(row.imagen)
            labels.append(get_label_by_emotion(row.clase))

        return images, labels

    def get_image_landmarks_and_labels_for_training(self):
        images = []
        labels = []

        if os.path.exists(FACESDB_LANDMARKS_FULL_PATH):
            faces_db = pd.read_csv(FACESDB_LANDMARKS_FULL_PATH)
            for index, row in faces_db.iterrows():
                images.append(ast.literal_eval(row.imagen))
                labels.append(get_label_by_emotion(row.clase))
        else:
            print("Se saltea "+FACESDB_LANDMARKS_FULL_PATH+" porque no existe")

        if os.path.exists(FACESGOOGLESETDB_LANDMARKS_FULL_PATH):
            faces_google_set_db = pd.read_csv(FACESGOOGLESETDB_LANDMARKS_FULL_PATH)
            for index, row in faces_google_set_db.iterrows():
                images.append(ast.literal_eval(row.imagen))
                labels.append(get_label_by_emotion(row.clase))
        else:
            print("Se saltea "+FACESGOOGLESETDB_LANDMARKS_FULL_PATH+" porque no existe")

        if os.path.exists(FERDB_LANDMARKS_FULL_PATH):
            fer_db = pd.read_csv(FERDB_LANDMARKS_FULL_PATH)
            for index, row in fer_db.iterrows():
                images.append(ast.literal_eval(row.imagen))
                labels.append(get_label_by_emotion(row.clase))
        else:
            print("Se saltea "+FERDB_LANDMARKS_FULL_PATH+" porque no existe")

        return images, labels

    @staticmethod
    def plot_confusion_matrix(cm, classes=EMOCIONES.values(), normalize=True, title='Confusion matrix',
                              cmap=plt.cm.Greens, archivo='plt_confusion_matrix.png'):
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
        plt.close()

    @staticmethod
    def plot_learning_curve(estimator, title, ylabel, xlabel, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
                            train_sizes=np.linspace(.1, 1.0, 5), archivo='plt_learning_curve.png'):

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        plt.style.use('seaborn')
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, test_scores_mean, label='Validation error')
        plt.ylabel(ylabel, fontsize=14)
        plt.xlabel(xlabel, fontsize=14)
        plt.title(title, fontsize=18, y=1.03)
        plt.legend()
        plt.savefig(archivo)
        plt.close()

    def plot_classification_report(self, test_labels, predicted_labels):
        report_dict = classification_report(test_labels, predicted_labels)
        # dataframe = pd.DataFrame.from_dict(report_dict)
        print(report_dict)
