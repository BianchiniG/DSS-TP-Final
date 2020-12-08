from modelos.RandomForest import RandomForest
from modelos.CNN import CNN
from Preprocesamiento import Preprocesamiento


class Reconocimiento:
    def __init__(self):
        self.rf = RandomForest()
        self.svm = None
        self.cnn = CNN()

    def ejecutar(self, file):
        p = Preprocesamiento()
        imagen_proc = p.preprocess_image(file)
        emocion_rf = self.rf.predict(imagen_proc)
        emocion_cnn = self.cnn.predict(imagen_proc)
        return {'rf': emocion_rf, 'svm': 'emotion_svm', 'cnn': emocion_cnn}

    def get_data(self):
        return {
            'rf': self.rf_data,
            'svm': self.svm_data,
            'cnn': self.cnn_data
        }
