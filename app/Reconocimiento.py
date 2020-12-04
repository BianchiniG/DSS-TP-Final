from modelos.RandomForest import RandomForest
# Demas imports a modelos


class Reconocimiento:
    def __init__(self):
        self.rf = RandomForest()
        self.svm = None
        self.cnn = None
        self.rf_data = {}
        self.svm_data = {}
        self.cnn_data = {}

    def ejecutar(self, frame):
        # self.rf_data = self.rf.predict(frame)
        # Demas llamadas a modelos
        pass

    def get_data(self):
        return {
            'rf': self.rf_data,
            'svm': self.svm_data,
            'cnn': self.cnn_data
        }
