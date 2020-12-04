DB_BASEPATH = '/app/datos/'
FACESDB_ROUTE = '/app/datos/datasets/faces-db/faces-db.csv'
FACESGOOGLESET_ROUTE = '/app/datos/datasets/faces-googleset/faces-googleset-db.csv'
LANDMARKS_SHAPE_PREDICTOR_FILE = '/app/datos/shape_predictor_68_face_landmarks.dat'

RF_TRAINED_MODEL_FILE = '/app/datos/trained/rf_trained_model.pkl'
CNN_TRAINED_MODEL_FILE = '/app/datos/trained/cnn_trained_model.h5'
SVM_TRAINED_MODEL_FILE = '/app/datos/trained/svm_trained_model.loquesea'

EMOCIONES = {
    0: 'anger',
    1: 'fear',
    2: 'happy',
    3: 'kiss',
    4: 'neutral',
    5: 'sadness',
    6: 'disgust',
    7: 'surprise'
}


def get_label_by_emotion(e):
    for label, emotion in EMOCIONES.items():
        if emotion == e:
            return label
    return None
