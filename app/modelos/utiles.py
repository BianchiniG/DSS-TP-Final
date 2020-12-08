DB_BASEPATH = '/app/datos/'

FACESDB_ROUTE = 'datasets/faces-db/'
FACESGOOGLESET_ROUTE = 'datasets/faces-googleset/'
FER_ROUTE = 'datasets/fer/'

FACESDB_FILE = 'faces-db.csv'
FACESGOOGLESET_FILE = 'faces-googleset-db.csv'
FER_FILE = 'fer.csv'

FACESDB_LANDMARKS_FILE = 'faces-db-landmarks.csv'
FACESGOOGLESET_LANDMARKS_FILE = 'faces-googleset-db-landmarks.csv'
FER_LANDMARKS_FILE = 'fer-landmarks.csv'

FACESDB_FULL_PATH = DB_BASEPATH+FACESDB_ROUTE+FACESDB_FILE
FACESGOOGLESETDB_FULL_PATH = DB_BASEPATH+FACESGOOGLESET_ROUTE+FACESGOOGLESET_FILE
FERDB_FULL_PATH = DB_BASEPATH+FER_ROUTE+FER_FILE

FACESDB_LANDMARKS_FULL_PATH = DB_BASEPATH+FACESDB_ROUTE+FACESDB_LANDMARKS_FILE
FACESGOOGLESETDB_LANDMARKS_FULL_PATH = DB_BASEPATH+FACESGOOGLESET_ROUTE+FACESGOOGLESET_LANDMARKS_FILE
FERDB_LANDMARKS_FULL_PATH = DB_BASEPATH+FER_ROUTE+FER_LANDMARKS_FILE

FACESDB_COMPRESSED_FILE = DB_BASEPATH+'datasets/faces-db.tar.xz'
FACESGOOGLESET_COMPRESSED_FILE = DB_BASEPATH+'datasets/faces-googleset.zip'
FER_ORIGINAL_FILE = DB_BASEPATH+'datasets/fer2013.csv'

LANDMARKS_SHAPE_PREDICTOR_FILE = '/app/datos/shape_predictor_68_face_landmarks.dat'

RF_TRAINED_MODEL_FILE = '/app/datos/trained/rf_trained_model.pkl'
CNN_TRAINED_MODEL_FILE_SAVE = '/app/datos/trained/CNN-TPU-1.h5'
SVM_TRAINED_MODEL_FILE = '/app/datos/trained/svm_trained_model.pkl'

CNN_TRAINED_MODEL_FILE_LOAD = '/app/datos/trained/CNN-TPU-1.h5'
TRAINED_CONFUSION_MATRIX_PLOT = '/app/static/img/CNN_fit_confusion_matrix9.png'
TRAINED_LEARNING_CURVE_PLOT = '/app/static/img/CNN_fit_learning_curve9.png'


EMOCIONES = {
    0: 'anger',
    1: 'fear',
    2: 'happy',
    3: 'neutral',
    4: 'sadness',
    5: 'disgust',
    6: 'surprise'
}
FER_DATASET_MAP = {
    0: 0,
    1: 5,
    2: 1,
    3: 2,
    4: 4,
    5: 6,
    6: 3
}

IMG_ROWS = 48
IMG_COLS = 48


def get_label_by_emotion(e):
    for label, emotion in EMOCIONES.items():
        if emotion == e:
            return label
    return None
