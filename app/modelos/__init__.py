from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier


class RandomForest:
    def __init__(self):
        pass

    def entrenar(self):
        pass

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    score = []
    print("### BAGGing+RandomForest")
    classifier = DecisionTreeClassifier().fit(X_train, y_train)
    print("#### Obtenemos el puntaje de ajuste del modelo...")
    print("- Para los datos de entrenamiento:", classifier.score(X_train, y_train))
    print("- Para los datos de prueba:", classifier.score(X_test, y_test))
    print("Se puede ver que tenemos overfitting en el puntaje de los datos de entrenamiento.\n")

    print("#### Utilizamos random forest como método de ensamble...")
    rf = RandomForestClassifier(n_estimators=100)
    bag_clf = BaggingClassifier(base_estimator=rf, n_estimators=100,
                                bootstrap=True, n_jobs=-1,
                                random_state=42)
    bag_clf.fit(X_train, y_train)
    print("### Obtenemos el puntaje de ajuste del modelo luego de aplicar Random Forest...")
    print("- Para los datos de entrenamiento:", bag_clf.score(X_train, y_train))
    print("- Para los datos de prueba:", bag_clf.score(X_test, y_test))
    print("Ya no tenemos más overfitting!\n")
    print("### Realizamos las predicciones: ")
    y_pred = bag_clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('#### Chequeamos utilizando la matriz de confusión para ver si el modelo no ha mezclado las clases:')
    print(confusion_matrix(y_test,y_pred))
