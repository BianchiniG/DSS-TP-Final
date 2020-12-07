
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from .utiles import EMOCIONES, CNN_TRAINED_MODEL_FILE
from .Model import Model
import random
from time import time
from keras.preprocessing.image import ImageDataGenerator

TRAINED_CONFUSION_MATRIX_PLOT = '/app/static/img/CNN_fit_confusion_matrix8.png'
TRAINED_LEARNING_CURVE_PLOT = '/app/static/img/CNN_fit_learning_curve8.png'

class CNN(Model):
    def fit(self):
        images, labels = self.get_images_and_labels_for_training()
        image_labels = self.__get_labeled_images(images, labels)
        
        train_data, test_data, train_label, test_label = self.__get_data_sets(image_labels)
        
        model, test_data, test_label, historia = self.__train_model(test_data, test_label, train_data, train_label)
        
        model.save(CNN_TRAINED_MODEL_FILE)
        
        print('Prediciendo CNN utilizando el set de prueba')
        confusion_matrix = self.__test_model(model, test_data, test_label)
        self.plot_confusion_matrix(cm=confusion_matrix, archivo=TRAINED_CONFUSION_MATRIX_PLOT)
        self.__plot_learning_curve(historia)
        

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
        train_data = []
        test_data = []
        train_label = []
        test_label = []

        start_time = time()
        train_image_labels = image_labels[:int(len(image_labels) * 0.8)]
        test_image_labels = image_labels[-int(len(image_labels) * 0.2):]

        print(len(train_image_labels))
        print(len(test_image_labels))
        for item in train_image_labels:
            image = cv2.imread(item[0],0)
            if len(image)>48:
                print(str(len(image))+' '+item[0])
            train_data.append(image)
            train_label.append(item[1])

        train_time = time() - start_time
        print('Dataset de entrenamiento extraído en %f segundos' % train_time)

        for item in test_image_labels:
            image = cv2.imread(item[0],0)
            if len(image)>48:
                print(str(len(image))+' '+item[0])
            test_data.append(image)
            test_label.append(item[1])

        print('Dataset de pruebas extraído en %f segundos' % (time() - train_time))

        train_data = np.array(train_data, dtype=np.uint8)
        train_data = np.array(train_data,'float32')
        test_data = np.array(test_data,'float32')

        train_data = train_data/255
        test_data = test_data/255       

        train_label=np_utils.to_categorical(train_label, num_classes=len(EMOCIONES))
        test_label=np_utils.to_categorical(test_label, num_classes=len(EMOCIONES))

        train_data = train_data.reshape(train_data.shape[0], 48, 48, 1)
        test_data = test_data.reshape(test_data.shape[0], 48, 48, 1)

        return train_data, test_data, train_label, test_label


    
    def __train_model(self, test_data, test_label, train_data, train_label):
        batch_size = 64
        epochs = 40

        start_time = time()
        
        model = Sequential()
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Dropout(0.25)) #para reducir el overfitting del modelo

        model.add(Conv2D(128, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten()) #se aplanan los datos
        model.add(Dense(256)) #se agrega una red convolucional
        model.add(Activation('elu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(EMOCIONES)))        #neuronas finales
        #model.add(Dense(len(EMOCIONES),activation='softmax'))        #neuronas finales
        model.add(Activation('softmax')) #para clasificar emociones entrantes
        #model.summary()

        #opt = SGD(lr=0.001, momentum=0.9, decay=0.01)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
        datagen = ImageDataGenerator(
                rotation_range = 5,
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                #brightness_range=[0.1,0.4],
                #rescale = 1./255,
                shear_range = 0.1,
                zoom_range = 0.1,
                #horizontal_flip = True,
                fill_mode = 'nearest')

        print(len(train_data))
        history = model.fit(train_data,train_label,validation_data=(test_data,test_label), batch_size=batch_size, epochs=epochs, verbose=1)
        #historia = model.fit(datagen.flow(train_data,train_label,shuffle=True),batch_size=batch_size,
        #                    validation_data=(test_data,test_label), epochs=100, verbose=1)
        
        print('El entrenamiento finalizó en %f segundos' % (time() - start_time))
        # self.plot_classification_report()
        
        #scores = model.evaluate(test_data, test_label, verbose=0)

        return model, test_data, test_label, historia

    def __plot_learning_curve(self, historia):
        accuracy = historia.history['accuracy']
        val_accuracy = historia.history['val_accuracy']
        loss = historia.history['loss']
        val_loss = historia.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(TRAINED_LEARNING_CURVE_PLOT)

    #def __convert_image(self, str_image):
    #    image = cv2.imread(str_image,0)
    #    image = cv2.resize(image, (48, 48))
    #    return image

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