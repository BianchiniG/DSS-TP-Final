from modelos.RandomForest import RandomForest
# from modelos.SVM import SVM
from modelos.CNN import CNN
from Preprocesamiento import Preprocesamiento


class Reconocimiento:
    def __init__(self):
        self.rf = RandomForest()
        self.svm = None  # SVM()
        self.cnn = CNN()

    def ejecutar(self, file):
        p = Preprocesamiento()
        imagen_proc = p.preprocess_image(file)

        emocion_rf = self.rf.predict(imagen_proc)
        emocion_svm = None  # self.svm.predict(imagen_proc)
        emocion_cnn = self.cnn.predict(imagen_proc)

        return {
            'rf': emocion_rf,
            'svm': emocion_svm,
            'cnn': emocion_cnn
        }
