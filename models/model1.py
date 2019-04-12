import os
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class LogRskLearn:

    def __init__( self ):
        self.lg = LogisticRegression(penalty='l2', multi_class='multinomial', solver='lbfgs')

    def train( self, xdata, ydata ):
        """Entrenamiento para un modelo multinomial de regularizacion L2 utilizando un metodo de optimizacion quasi-Newton"""

        print("Entrenando el modelo")
        self.lg.fit(xdata, ydata)
        joblib.dump(self.lg, './models/model1/saved/LogRskLearn.pkl')
        print("Modelo guardado en: ", os.path.abspath('/models/model1/saved/LogRskLearn.pkl'))


    def predict( self, data):
        self.lg = joblib.load(os.path.abspath('models/model1/saved/LogRskLearn.pkl'))
        return self.lg.predict(data)


    def accuracy(self, X_data, Y_data ):
        self.lg = joblib.load(os.path.abspath('models/model1/saved/LogRskLearn.pkl'))
        y_modelo = self.lg.predict(X_data)
        accuracy = accuracy_score(Y_data, y_modelo) * 100
        print('Precision obtenida {:.2f}%'.format(round(accuracy, 2)))
