import abc


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self):
        """
        Acá entrenamos el modelo con los datos preprocesados (Se implementa en cada clase concreta).
        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, image):
        """
        Acá, en base al modelo ya entrenado, predecimos qué emocion contiene la cara de la imagen que nos mandan (Se
        implementa en cada clase concreta).
        :param image:
        :return:
        """
        pass
