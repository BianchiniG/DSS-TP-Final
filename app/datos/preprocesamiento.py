import cv2
import os
from pathlib import Path
import pandas as pd

FACESDB_FOLDER = 'datasets/faces-db'
FACESGOOGLESET_FOLDER = 'datasets/faces-googleset'
FACESDB_OUT_FILENAME = 'faces-db.csv'
FACESGOOGLESET_OUT_FILENAME = 'faces-googleset-db.csv'
EMOCIONES = {
    'anger': 'anger',
    'fear': 'fear',
    'joy': 'happy',
    'happy': 'happy',
    'kiss': 'kiss',
    'neutral': 'neutral',
    'sadness': 'sadness',
    'disgust': 'disgust',
    'surprise': 'surprise'
}
facesdb_folder = Path(FACESDB_FOLDER)
facesgoogleset_folder = Path(FACESGOOGLESET_FOLDER)


def generate_db(folder, dataframe):
    for file in os.scandir(folder):
        if os.path.isdir(file):
            generate_db(file, dataframe)
        else:
            index = dataframe.count().imagen + 1
            emocion = [EMOCIONES[e] for e in EMOCIONES.keys() if e in file.path][0]
            datos = {'index': index, 'imagen': file.path, 'clase': emocion}
            dataframe.loc[index] = datos


def preprocesar(data_folder, db_file):
    data_folder = Path(data_folder)
    df = pd.read_csv(db_file)

    for index, row in df.iterrows():
        file_to_open = data_folder / row.imagen
        print(file_to_open)
        img = cv2.imread(os.path.join(file_to_open),cv2.IMREAD_GRAYSCALE)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5)
        (x, y, w, h) = faces_detected[0] 
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        imgr = img[y:y+h, x:x+w]
        nuevo = cv2.resize(imgr, (480, 640))#(I1.shape[1],I1.shape[0]))
        r = cv2.imwrite(os.path.join(data_folder, str(row.imagen)), nuevo)


def create_dbs():
    dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
    generate_db(facesdb_folder, dataframe)
    dataframe.to_csv(FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME)
    dataframe = pd.DataFrame(data=[], columns=['index', 'imagen', 'clase'])
    generate_db(facesgoogleset_folder, dataframe)
    dataframe.to_csv(FACESGOOGLESET_FOLDER+'/'+FACESGOOGLESET_OUT_FILENAME)


# create_dbs()
preprocesar(FACESDB_FOLDER, FACESDB_FOLDER+'/'+FACESDB_OUT_FILENAME)
# preprocesar(FACESGOOGLESET_FOLDER, FACESGOOGLESET_OUT_FILENAME)
