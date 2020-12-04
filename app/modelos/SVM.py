import cv2, random, math
import numpy as np
import dlib, os, fnmatch, operator
from sklearn.svm import SVC
from modelos.Model import Model
from modelos.utiles import EMOCIONES, SVM_TRAINED_MODEL_FILE, LANDMARKS_SHAPE_PREDICTOR_FILE
from time import time
from sklearn.metrics import confusion_matrix
import joblib

TRAINED_CONFUSION_MATRIX_PLOT = '/app/static/img/svm_fit_confusion_matrix.png'

class SVM(Model):
        
    def fit(self):
        accur_lin = []
        images, labels = self.get_images_and_labels_for_training()
        image_labels = self.__get_labeled_images(images, labels)
        print("Cantidad de imagenes", len(images))
        # Convierte el conjunto de entrenamiento en una matriz numpy para el clasificador
        train_data, train_label, test_data, test_label = self.__get_data_sets(image_labels)
        print('Prediciendo el svm utilizando el set de prueba')
        model = self.__train_model(train_data, train_label)
        joblib.dump(model, SVM_TRAINED_MODEL_FILE, compress=9)
        print('Prediciendo random forest utilizando el set de prueba')
        confusion_matrix = self.__test_model(model, test_data, test_label)
        self.plot_confusion_matrix(cm=confusion_matrix, archivo=TRAINED_CONFUSION_MATRIX_PLOT)

    def __train_model(self, train_data, train_label):
        start_time = time()
        #Establecer SVM
        clf = SVC(kernel='linear', probability=True, tol=1e-3)
        npar_train = np.array(train_data) 
        npar_trainlabs = np.array(train_label)
        # Entrenamiento SVM
        print("Entrenando SVM") 
        clf.fit(npar_train, train_label)
        print('El entrenamiento finalizó en %f segundos' % (time() - start_time))
        return clf 

    def __test_model(self, clf, test_data, test_label):
        accur_lin = []
        start_time = time()
        # Utiliza la función score () para obtener mayor precisión
        print("Obteniendo precision") 
        npar_pred = np.array(test_data)
        npar_predlabs = np.array(test_label)
        pred_lin = clf.score(npar_pred, npar_predlabs)
        # Guarda la precision en una lista
        print ("linear: ", pred_lin)
        accur_lin.append(pred_lin) 
        proba = clf.predict(test_data)
        #print ("proba: ", proba)
        print('La predicción finalizó en %f segundos' % (time() - start_time))
        return confusion_matrix(test_label, proba)
    
    def __get_labeled_images(self, images, labels):
        image_labels = list(zip(images, labels))
        random.seed(time())
        random.shuffle(image_labels)
        return image_labels

    def predict(self, image):
        model = joblib.load(SVM_TRAINED_MODEL_FILE)
        image_landmarks = self.get_landmarks(cv2.imread(image))
        if image_landmarks != "error":   
            prediction = model.predict(np.array(image_landmarks).reshape(1, -1))
            print("Emocion ", EMOCIONES[prediction[0]])
            return EMOCIONES[prediction[0]]        
        else: 
            return "error"

    def get_landmarks(self, image):
        # inicializa el detector facial de dlib (basado en HOG) y luego crea
        # el predictor de hitos faciales
        detector = dlib.get_frontal_face_detector()    
        predictor = dlib.shape_predictor(LANDMARKS_SHAPE_PREDICTOR_FILE )
        detections = detector(image, 1)
        # Para todas las cara detectadas de forma individual
        for k, d in enumerate(detections):
            # Determina los puntos de referencia faciales para la región de la cara, luego
            shape = predictor(image, d) 
            xlist = []
            ylist = []
            
            # Guarda coordenadas X e Y en dos listas
            for i in range(1, 68): 
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
            
            # Obtiene la media de ambos ejes para determinar el centro de gravedad 
            xmean = np.mean(xlist) 
            ymean = np.mean(ylist)
            # Calcula distancia entre cada punto y el punto central en ambos ejes
            xcentral = [(x-xmean) for x in xlist] 
            ycentral = [(y-ymean) for y in ylist]

            # Si la coordenada x del conjunto son las mismas, el ángulo es 0,  evitamos el error 'divide by 0' en la función
            # Porque en el eje x el [26] seria la mitad de la ceja y el [29] seria la mitad de la nariz
            if xlist[26] == xlist[29]: 
                anglenose = 0
            else:
                anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi) 
            # Le suma o resta 90 para recorrer los cuadrantes 
            if anglenose < 0:
                anglenose += 90
            else:
                anglenose -= 90

            landmarks_vectorised = []
            
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                # Guarda en el arreglo landmarks vectorizado la distancia al centro de cada x, y
                landmarks_vectorised.append(x)
                landmarks_vectorised.append(y)
                # Crea un arreglo con la media x, media y 
                meannp = np.asarray((ymean, xmean))
                # Crea un arreglo con la coordenada x y la coordenada y
                coornp = np.asarray((z, w))
                # Retorna un vector normalizado
                dist = np.linalg.norm(coornp-meannp)
                # Calcula el angulo relativo, restandole al arcotangente el angulo de la nariz calculado anteriormente
                anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append(anglerelative)

        if len(detections) < 1: 
            landmarks_vectorised = "error"
        # Retorna el arreglo de marcas vectorizadas que contiene el xcentral, ycentral, el vector normalizado y angulo relativo
        return landmarks_vectorised

    def __get_data_sets(self, image_labels):
        train_data = []
        train_labels = []
        test_data =[]
        test_labels =[]
        start_time = time()
        train_images = image_labels[:int(len(image_labels) * 0.8)]
        test_images = image_labels[-int(len(image_labels) * 0.2):]
        print("Cantidad de imagenes de entrenamiento", len(train_images))
        print("Cantidad de imagenes de entrenamiento", len(test_images))
        # Genera etiquetas para la lista de entramiento 
        for item in train_images:
            # Abre la imagen
            image = cv2.imread(item[0]) 
            landmarks_vectorised = self.get_landmarks(image)
            if landmarks_vectorised == "error":
                print("ERROR imagen de train", item[0])
            else:
                # Vector de imágenes a la lista de datos de entrenamiento
                train_data.append(landmarks_vectorised) 
                # Etiqueta de emocion
                train_labels.append(item[1])
                #print("vectorizada la imagen de entrenamiento", train_images.index(item))
        
        train_time = time() - start_time
        print('Dataset de entrenamiento extraído en %f segundos' % train_time)
        
        for item in test_images:
            image = cv2.imread(item[0]) 
            landmarks_vectorised = self.get_landmarks(image)
            if landmarks_vectorised == "error":
                print("ERROR imagen testeo ", item[0])
            else:
                # Vector de imágenes a la lista de datos de entrenamiento
                test_data.append(landmarks_vectorised) 
                # Etiqueta de emocion
                test_labels.append(item[1])
                #print("vectorizada la imagen de testeo", test_images.index(item))

        print('Dataset de pruebas extraído en %f segundos' % (time() - train_time))
    
        return train_data, train_labels, test_data, test_labels
    