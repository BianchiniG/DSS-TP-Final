import cv2, random, math
import numpy as np
import dlib, os, fnmatch, operator
from sklearn.svm import SVC
from modelos.Model import Model
from modelos.utiles import EMOCIONES, SVM_TRAINED_MODEL_FILE, LANDMARKS_SHAPE_PREDICTOR_FILE
from time import time
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek 

TRAINED_CONFUSION_MATRIX_PLOT = '/app/static/img/svm_fit_confusion_matrix.png'
TRAINED_LEARNING_CURVE_PLOT = '/app/static/img/svm_fit_learning_curve.png'

class SVM(Model):
        
    def fit(self):
        accur_lin = []
        images, labels = self.get_image_landmarks_and_labels_for_training()
        print("Cantidad de imagenes", len(images))
        image_labels = self.__get_labeled_images(images, labels)
        # Convierte el conjunto de entrenamiento en una matriz numpy para el clasificador
        print('Extrayendo landmarks de las imagenes y dividiendo el dataset en entrenamiento/pruebas')
        train_data, train_label, test_data, test_label = self.__get_data_sets(image_labels)
        model, test_data, test_label = self.__train_model(train_data, train_label, test_data, test_label)
        joblib.dump(model, 'SVM_TRAINED_MODEL_FILE, compress=9)
        print('Prediciendo SVM utilizando el set de prueba')
        confusion_matrix = self.__test_model(model, test_data, test_label)
        self.plot_confusion_matrix(cm=confusion_matrix, archivo=TRAINED_CONFUSION_MATRIX_PLOT)

    def predict(self, image):
        preprocesador = Preprocesamiento()
        model = joblib.load(SVM_TRAINED_MODEL_FILE)
        prediction = model.predict(np.array(preprocesador.get_landmarks(preprocesador.preprocess_image(image))).reshape(1, -1))
        print("Emocion ", EMOCIONES[prediction[0]])
        return EMOCIONES[prediction[0]]        
        
    def __train_model(self, train_data, train_label, test_data, test_label):
        start_time = time()
        # Establecer SVM
        clf = SVC(kernel='rbf')
        train_data = np.asarray(train_data) 
        train_label = np.asarray(train_label)
        test_data = np.asarray(test_data)
        test_label = np.asarray(test_label)
        # Entrenamiento SVM
        print("Entrenando SVM") 
        clf.fit(train_data, train_label)
        print('El entrenamiento finalizó en %f segundos' % (time() - start_time))
        self.plot_learning_curve(SVC(kernel = 'rbf'), 'Curva de Aprendizaje SVM',
                                    'Tamaño del set de pruebas', 'Errores', test_data, test_label,
                                    archivo=TRAINED_LEARNING_CURVE_PLOT)
        return clf, test_data, test_label

    def __test_model(self, clf, test_data, test_label):
        accur_lin = []
        start_time = time()
        # Utiliza la función score () para obtener mayor precisión
        pred_lin = clf.score(test_data, test_label)
        svm_predict_labels = clf.predict(test_data)
        #print ("proba: ", proba)
        print('La predicción finalizó en %f segundos' % (time() - start_time))
        print('Precisión del entrenamiento: %f' % (np.mean(svm_predict_labels == test_label) * 100))
        self.plot_classification_report(test_label,svm_predict_labels)
        return confusion_matrix(test_label, svm_predict_labels)
    
    def __get_labeled_images(self, images, labels):
        us = SMOTETomek()  
        image_labels = list(zip(images, labels))
        random.seed(time())
        random.shuffle(image_labels)
        return image_labels

    def __get_data_sets(self, image_labels):
        train_data = []
        train_labels = []
        test_data =[]
        test_labels =[]
        train_images = image_labels[:int(len(image_labels) * 0.8)]
        test_images = image_labels[-int(len(image_labels) * 0.2):]
        print("Cantidad de imagenes de entrenamiento", len(train_images))
        print("Cantidad de imagenes de entrenamiento", len(test_images))
        # Genera etiquetas para la lista de entramiento 
        for item in train_images:
            # Vector de imágenes a la lista de datos de entrenamiento
            train_data.append(item[0]) 
            # Etiqueta de emocion
            train_labels.append(item[1])
        
        for item in test_images:
            # Vector de imágenes a la lista de datos de entrenamiento
            test_data.append(item[0]) 
            # Etiqueta de emocion
            test_labels.append(item[1])
            
        return train_data, train_labels, test_data, test_labels
    