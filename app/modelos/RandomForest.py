import sys
sys.path.append("..")
import joblib
import random
import numpy as np
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from Preprocesamiento import Preprocesamiento
from .Model import Model
from .utiles import EMOCIONES, RF_TRAINED_MODEL_FILE

N_ESTIMATORS = 100
BETWEEN_EYES_LANDMARK = 26
NOSE_TIP_LANDMARK = 29
TRAINED_CONFUSION_MATRIX_PLOT = '/app/static/img/random_forest_fit_confusion_matrix.png'
TRAINED_LEARNING_CURVE_PLOT = '/app/static/img/random_forest_fit_learning_curve.png'


class RandomForest(Model):
    def fit(self):
        images, labels = self.get_image_landmarks_and_labels_for_training()
        image_labels = self.__get_labeled_images(images, labels)
        print('Extrayendo landmarks de las imagenes y dividiendo el dataset en entrenamiento/pruebas')
        train_data, test_data, train_label, test_label = self.__get_data_sets(image_labels)
        print('Entrenando random forest con %d estimadores' % N_ESTIMATORS)
        model, test_data, test_label = self.__train_model(test_data, test_label, train_data, train_label)
        joblib.dump(model, RF_TRAINED_MODEL_FILE, compress=9)
        print('Prediciendo random forest utilizando el set de prueba')
        confusion_matrix = self.__test_model(model, test_data, test_label)
        self.plot_confusion_matrix(cm=confusion_matrix, archivo=TRAINED_CONFUSION_MATRIX_PLOT)

    def predict(self, image):
        preprocesador = Preprocesamiento()
        model = joblib.load(RF_TRAINED_MODEL_FILE)
        prediction = model.predict(np.array(preprocesador.get_landmarks(image)).reshape(1, -1))
        return EMOCIONES[prediction[0]]

    def __train_model(self, test_data, test_label, train_data, train_label):
        start_time = time()
        rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                          max_features=6,
                                          random_state=42,
                                          criterion='entropy')
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label)
        test_data = np.asarray(test_data)
        test_label = np.asarray(test_label)
        rf_model.fit(train_data, train_label)
        # print("Obtenemos el puntaje de ajuste del modelo...")
        # print("- Para los datos de entrenamiento:", model.score(train_data, train_label))
        # print("- Para los datos de prueba:", model.score(test_data, test_label))
        print('El entrenamiento finalizó en %f segundos' % (time() - start_time))
        # self.plot_classification_report()
        self.plot_learning_curve(RandomForestRegressor(), 'Curva de Aprendizaje usando regresor Random Forest',
                                 'Tamaño del set de pruebas', 'Errores', test_data, test_label,
                                 archivo=TRAINED_LEARNING_CURVE_PLOT)
        return rf_model, test_data, test_label

    def __test_model(self, rf_clf, test_data, test_label):
        start_time = time()
        rf_predicted_labels = rf_clf.predict(test_data)
        print('La predicción finalizó en %f segundos' % (time() - start_time))
        print('Precisión del entrenamiento: %f' % (np.mean(rf_predicted_labels == test_label) * 100))
        return confusion_matrix(test_label, rf_predicted_labels)

    def __get_labeled_images(self, images, labels):
        image_labels = list(zip(images, labels))
        random.seed(time())
        random.shuffle(image_labels)
        return image_labels

    def __get_data_sets(self, image_labels):
        train_data = []
        test_data = []
        train_label = []
        test_label = []
        train_image_labels = image_labels[:int(len(image_labels) * 0.8)]
        test_image_labels = image_labels[-int(len(image_labels) * 0.2):]

        for item in train_image_labels:
            train_data.append(item[0])
            train_label.append(item[1])
        for item in test_image_labels:
            test_data.append(item[0])
            test_label.append(item[1])

        return train_data, test_data, train_label, test_label
