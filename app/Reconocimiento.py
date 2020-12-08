from modelos.RandomForest import RandomForest
from modelos.CNN import CNN
# Demas imports a modelos
from Preprocesamiento import Preprocesamiento

class Reconocimiento:
    def __init__(self):
        self.rf = RandomForest()
        self.svm = None
        self.cnn = CNN()
        self.rf_data = {}
        self.svm_data = {}
        self.cnn_data = {}

    def ejecutar(self, file):
        # self.rf_data = self.rf.predict(frame)
        # Demas llamadas a modelos
        p = Preprocesamiento()
        imagen_proc = p.preprocess_image(file)
        emocion_cnn = self.cnn.predict(imagen_proc)
        return {'rf':'emotion_rf','svm':'emotion_svm','cnn':emocion_cnn}

    def get_data(self):
        return {
            'rf': self.rf_data,
            'svm': self.svm_data,
            'cnn': self.cnn_data
        }
