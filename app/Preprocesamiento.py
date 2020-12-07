import cv2
import os
import csv
import cv2
import dlib
import imageio
import multiprocessing
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from modelos.utiles import DB_BASEPATH, \
    IMG_COLS, \
    IMG_ROWS, \
    EMOCIONES, \
    FER_DATASET_MAP, \
    EMOCIONES, \
    LANDMARKS_SHAPE_PREDICTOR_FILE, \
    FACESDB_FULL_PATH, \
    FACESGOOGLESETDB_FULL_PATH, \
    FERDB_FULL_PATH, \
    FACESDB_LANDMARKS_FULL_PATH, \
    FACESGOOGLESETDB_LANDMARKS_FULL_PATH, \
    FERDB_LANDMARKS_FULL_PATH, \
    FER_ORIGINAL_FILE

BETWEEN_EYES_LANDMARK = 26
NOSE_TIP_LANDMARK = 29
FER_FOLDER = 'datasets/fer'
FACESDB_FOLDER = 'datasets/faces-db'
FACESGOOGLESET_FOLDER = 'datasets/faces-googleset'
FER_OUT_FILENAME = 'fer.csv'
FACESDB_OUT_FILENAME = 'faces-db.csv'
FACESGOOGLESET_OUT_FILENAME = 'faces-googleset-db.csv'
FACESLANDMARKSDB_OUT_FILENAME = 'faces-landmarks-db.csv'
GOOGLESLANDMARKSDB_OUT_FILENAME = 'faces-googleset-landmarks-db.csv'
FERLANDMARKSDB_OUT_FILENAME = 'fer-landmarks-db.csv'
EMOCIONES_MAP = {
    'anger': 'anger',
    'fear': 'fear',
    'joy': 'happy',
    'happy': 'happy',
    'neutral': 'neutral',
    'sadness': 'sadness',
    'disgust': 'disgust',
    'surprise': 'surprise'
}
LANDMARKS_DB_MAP = [
    {
        'landmark_db_file': FACESDB_LANDMARKS_FULL_PATH,
        'imagepath_db_file': FACESDB_FULL_PATH
    },
    {
        'landmark_db_file': FACESGOOGLESETDB_LANDMARKS_FULL_PATH,
        'imagepath_db_file': FACESGOOGLESETDB_FULL_PATH
    },
    {
        'landmark_db_file': FERDB_LANDMARKS_FULL_PATH,
        'imagepath_db_file': FERDB_FULL_PATH
    }
]


class Preprocesamiento:
    def __init__(self):
        self.facesdb_folder = Path(DB_BASEPATH+'/'+FACESDB_FOLDER)
        self.facesgoogleset_folder = Path(DB_BASEPATH+'/'+FACESGOOGLESET_FOLDER)
        self.fer_folder = Path(DB_BASEPATH+'/'+FER_FOLDER)

    @staticmethod
    def preprocess_image(imagen):
        if isinstance(imagen, str):
            imagen = cv2.imread(imagen)
        grayed_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_detected = face_cascade.detectMultiScale(grayed_img, scaleFactor=1.5, minNeighbors=5)
        if len(faces_detected):
            (x, y, w, h) = faces_detected[0]
            cv2.rectangle(grayed_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            img = cv2.normalize(grayed_img, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.resize(img[y:y+h, x:x+w], (IMG_COLS, IMG_ROWS))
        else:
            raise Exception()

    def preprocess_databases(self):
        if self.__databases_not_created():
            self.__create_dbs()

        databases = [DB_BASEPATH+FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME,
                     DB_BASEPATH+FACESGOOGLESET_FOLDER+'/'+FACESGOOGLESET_OUT_FILENAME]
        for database in databases:
            df = pd.read_csv(database)

            for index, row in df.iterrows():
                try:
                    imagen = cv2.imread(os.path.join(DB_BASEPATH, row.imagen))
                    cv2.imwrite(os.path.join(DB_BASEPATH, str(row.imagen)), self.preprocess_image(imagen))
                except Exception:
                    print("No se detecto una cara en la foto: "+os.path.join(DB_BASEPATH, row.imagen))
                    os.remove(os.path.join(DB_BASEPATH, row.imagen))
                    df = df.drop(index)
            os.remove(database)
            df.to_csv(database)



    def __create_dbs(self):
        dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
        self.__generate_db(self.facesdb_folder, dataframe)
        dataframe.to_csv(DB_BASEPATH+'/'+FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME)
        dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
        self.__generate_db(self.facesgoogleset_folder, dataframe)
        dataframe.to_csv(DB_BASEPATH+'/'+FACESGOOGLESET_FOLDER+'/'+FACESGOOGLESET_OUT_FILENAME)
        dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
        self.__generate_db(self.fer_folder, dataframe)
        dataframe.to_csv(DB_BASEPATH+'/'+FER_FOLDER+'/'+FER_OUT_FILENAME)

    def __generate_db(self, folder, dataframe):
        for file in os.scandir(folder):
            if os.path.isdir(file) and file != 'kiss':
                self.__generate_db(file, dataframe)
            else:
                _, extension = os.path.splitext(file)
                if not extension == '.csv':
                    index = dataframe.count().imagen + 1
                    emocion = [EMOCIONES_MAP[e] for e in EMOCIONES_MAP.keys() if e in file.path][0]
                    datos = {'index': index, 'imagen': file.path, 'clase': emocion}
                    dataframe.loc[index] = datos

    def export_fer_dataset(self, ruta=FER_ORIGINAL_FILE):
        with open(ruta, 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            headers = next(datareader)

            id = 1
            for row in datareader:
                if row[2] == 'Training':
                    emotion = EMOCIONES[FER_DATASET_MAP[int(row[0])]]
                    pixels = list(map(int, row[1].split()))
                    print("Exportando imagen "+str(id)+" - emocion "+str(emotion))

                    pixels_array = np.asarray(pixels)

                    image = pixels_array.reshape(IMG_ROWS, IMG_COLS)
                    image = image.astype(np.uint8)
                    image_folder = os.path.join(DB_BASEPATH+'/'+FER_FOLDER, emotion)
                    if not os.path.exists(image_folder):
                        os.mkdir(image_folder)
                    image_file = os.path.join(image_folder, str(id) + '.jpg')
                    imageio.imwrite(image_file, image)
                    id += 1

    def generate_landmarks_dbs(self):
        cpu_count = multiprocessing.cpu_count()
        for db in LANDMARKS_DB_MAP:
            with open(db['imagepath_db_file'], 'r') as db_file:
                print("Procesando "+db['imagepath_db_file'])
                start_time = time()
                df = pd.read_csv(db_file)
                image_label = list(zip(df.imagen, df.clase))
                pool = multiprocessing.Pool(cpu_count)
                extracted = pool.map(self.process_image_with_landmarks, image_label)
                extracted_with_face_detection = list(filter(lambda x: x[0] is not None and x[1] is not None, extracted))
                print('Ejecución finalizada en %f segundos' % (time() - start_time))
                dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
                for i, row in enumerate(extracted_with_face_detection):
                    dataframe.loc[i] = {'index': i, 'imagen': row[0], 'clase': row[1]}
                dataframe.to_csv(db['landmark_db_file'])
                print(str(len(extracted_with_face_detection))+" imagenes con detecciones extraídas del dataset "+db['imagepath_db_file']+" (De "+str(len(extracted))+")")

    def process_image_with_landmarks(self, image_label):
        emocion = image_label[1]
        imagen = cv2.imread(image_label[0])

        landmarks = self.get_landmarks(imagen)
        if landmarks is None:
            emocion = None

        return [landmarks, emocion]

    def get_landmarks(self, imagen):
        if isinstance(imagen, str):
            imagen = cv2.imread(imagen)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(LANDMARKS_SHAPE_PREDICTOR_FILE)
        image = imagen.copy()
        detections = detector(image, 1)

        if len(detections) < 1:
            return None

        shape = predictor(image, detections[0])
        x_list = []
        y_list = []
        for i in range(1, 68):
            x_list.append(float(shape.part(i).x))
            y_list.append(float(shape.part(i).y))

        x_mean = x_list[NOSE_TIP_LANDMARK]
        y_mean = y_list[NOSE_TIP_LANDMARK]
        x_central = [(x - x_mean) for x in x_list]
        y_central = [(y - y_mean) for y in y_list]

        angle_nose = np.arctan2(
            (y_list[BETWEEN_EYES_LANDMARK] - y_mean), (x_list[BETWEEN_EYES_LANDMARK] - x_mean)
        ) * 180 / np.pi
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks_vectorised = []
        for i in range(0, 67):
            rx = x_central[i]
            ry = y_central[i]
            x = x_list[i]
            y = y_list[i]
            landmarks_vectorised.append(rx)
            landmarks_vectorised.append(ry)
            dist = np.linalg.norm(np.array([rx, ry]))
            landmarks_vectorised.append(dist)
            angle_relative = (np.arctan2((-ry), (-rx)) * 180 / np.pi) - angle_nose
            landmarks_vectorised.append(angle_relative)

        return landmarks_vectorised

    @staticmethod
    def __databases_not_created():
        return not os.path.exists(DB_BASEPATH+'/'+FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME) or not \
            os.path.exists(DB_BASEPATH+'/'+FACESGOOGLESET_FOLDER + '/' + FACESGOOGLESET_OUT_FILENAME) or not \
            os.path.exists(DB_BASEPATH+'/'+FER_FOLDER+'/'+FER_OUT_FILENAME)
