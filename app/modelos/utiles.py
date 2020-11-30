DB_BASEPATH = '/app/datos/'
FACESDB_ROUTE = '/app/datos/datasets/faces-db/faces-db.csv'
FACESGOOGLESET_ROUTE = '/app/datos/datasets/faces-googleset/faces-googleset-db.csv'

RF_TRAINED_MODEL_FILE = '/app/datos/trained/rf_trained_model.loquesea'
CNN_TRAINED_MODEL_FILE = '/app/datos/trained/cnn_trained_model.loquesea'
SVM_TRAINED_MODEL_FILE = '/app/datos/trained/svm_trained_model.loquesea'

EMOCIONES = {
    0: 'anger',
    1: 'fear',
    2: 'happy',
    4: 'kiss',
    5: 'neutral',
    6: 'sadness',
    7: 'disgust',
    8: 'surprise'
}


def get_label_by_emotion(e):
    for label, emotion in EMOCIONES.items():
        if emotion == e:
            return label
    return None
