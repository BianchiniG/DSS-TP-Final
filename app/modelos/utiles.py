DB_BASEPATH = '/app/datos/'
FACESDB_ROUTE = '/app/datos/datasets/faces-db/faces-db.csv'
FACESGOOGLESET_ROUTE = '/app/datos/datasets/faces-googleset/faces-googleset-db.csv'
FER_ROUTE = '/app/datos/datasets/fer/fer.csv'
LANDMARKS_SHAPE_PREDICTOR_FILE = '/app/datos/shape_predictor_68_face_landmarks.dat'

RF_TRAINED_MODEL_FILE = '/app/datos/trained/rf_trained_model.pkl'
CNN_TRAINED_MODEL_FILE = '/app/datos/trained/cnn_trained_model.h5'
SVM_TRAINED_MODEL_FILE = '/app/datos/trained/svm_trained_model.pkl'

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
