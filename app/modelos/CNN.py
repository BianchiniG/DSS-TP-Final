
import cv2
import numpy as np

from keras.models import load_model
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from utiles import EMOCIONES, CNN_TRAINED_MODEL_FILE
from Model import Model
import random
from time import time

TRAINED_CONFUSION_MATRIX_PLOT = '/app/static/img/CNN_fit_confusion_matrix.png'

class CNN(Model):
    def fit(self):
        images, labels = self.get_images_and_labels_for_training()
        image_labels = self.__get_labeled_images(images, labels)
        
        print('Extrayendo landmarks de las imagenes y dividiendo el dataset en entrenamiento/pruebas')
        train_data, test_data, train_label, test_label = self.__get_data_sets(image_labels)
        
        model, test_data, test_label = self.__train_model(test_data, test_label, train_data, train_label)
        
        model.save(CNN_TRAINED_MODEL_FILE)
        
        print('Prediciendo CNN utilizando el set de prueba')
        confusion_matrix = self.__test_model(model, test_data, test_label)
        self.plot_confusion_matrix(cm=confusion_matrix, archivo=TRAINED_CONFUSION_MATRIX_PLOT)

    def __test_model(self, cnn_clf, test_data, test_label):
        start_time = time()
        cnn_predicted_labels = cnn_clf.predict(test_data)
        print('La predicción finalizó en %f segundos' % (time() - start_time))

        predicted_labels=[]
        for p in cnn_predicted_labels:
            predicted_labels.append(np.argmax(p))
        t_labels=[]
        for tl in test_label:
            t_labels.append(np.argmax(tl))

        return confusion_matrix(t_labels, predicted_labels)

    def __get_labeled_images(self, images, labels):
        image_labels = list(zip(images, labels))
        random.seed(time())
        random.shuffle(image_labels)
        return image_labels

    def predict(self, image):
        model = load_model(CNN_TRAINED_MODEL_FILE)
        prediction = model.predict(image)
        return EMOCIONES[np.argmax(prediction[0])]


    def __get_data_sets(self, image_labels):
        #num_labels = 7

        train_data = []
        test_data = []
        train_label = []
        test_label = []

        start_time = time()
        train_image_labels = image_labels[:int(len(image_labels) * 0.8)]
        test_image_labels = image_labels[-int(len(image_labels) * 0.2):]

        for item in train_image_labels:
            train_data.append(self.__convert_image(item[0]))
            train_label.append(item[1])

        train_time = time() - start_time
        print('Dataset de entrenamiento extraído en %f segundos' % train_time)

        for item in test_image_labels:
            test_data.append(self.__convert_image(item[0]))
            test_label.append(item[1])

        print('Dataset de pruebas extraído en %f segundos' % (time() - train_time))

        train_data = np.array(train_data, dtype=np.uint8)
        train_data = np.array(train_data,'float32')
        test_data = np.array(test_data,'float32')

        print("LLEGUEEEEEEEEEEEEEEEE ")
        train_data = train_data/255
        test_data = test_data/255       

        train_label=np_utils.to_categorical(train_label, num_classes=len(EMOCIONES))
        test_label=np_utils.to_categorical(test_label, num_classes=len(EMOCIONES))

        train_data = train_data.reshape(train_data.shape[0], 48, 48, 1)
        test_data = test_data.reshape(test_data.shape[0], 48, 48, 1)

        print(train_data.shape)
        print(test_data.shape)
        print(train_label.shape)
        print(test_label.shape)

        return train_data, test_data, train_label, test_label


    
    def __train_model(self, test_data, test_label, train_data, train_label):
        batch_size = 64
        epochs = 100

        start_time = time()
        
        model = Sequential()
        #model.add(BatchNormalization()) #facilita la convergencia del entrenamiento
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Dropout(0.25)) #para reducir el overfitting del modelo

        #model.add(BatchNormalization())
        model.add(Conv2D(128, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        #model.add(BatchNormalization())
        model.add(Conv2D(256, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten()) #se aplanan los datos
        model.add(Dense(256)) #se agrega una red convolucional
        model.add(Activation('elu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(EMOCIONES)))        #neuronas finales
        model.add(Activation('softmax')) #para clasificar emociones entrantes
        #model.summary()

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

        #train_data = np.asarray(train_data)
        #train_label = np.asarray(train_label)
        #test_data = np.asarray(test_data)
        #test_label = np.asarray(test_label)

        history = model.fit(train_data,train_label,validation_data=(test_data,test_label), batch_size=batch_size, epochs=epochs, verbose=1)

        # print("Obtenemos el puntaje de ajuste del modelo...")
        # print("- Para los datos de entrenamiento:", model.score(train_data, train_label))
        # print("- Para los datos de prueba:", model.score(test_data, test_label))
        print('El entrenamiento finalizó en %f segundos' % (time() - start_time))
        # self.plot_classification_report()

        return model, test_data, test_label


    def __convert_image(self, str_image):
        image = cv2.imread(str_image,0)
        image = cv2.resize(image, (48, 48))
        return image

    #'../datos/datasets/faces-googleset/happy/google_016.jpg'
    def cargar_imagen(self,ruta):
        imagen = cv2.imread(ruta,0)
        imagen = cv2.resize(imagen, (48, 48))
        images=[]
        images.append(imagen)
        X_images = np.array(images, dtype=np.uint8)
        test_X_images = X_images.astype('float32')
        test_X_images = test_X_images / 255
        test_X_images = test_X_images.reshape(test_X_images.shape[0], 48, 48, 1)
        return test_X_images