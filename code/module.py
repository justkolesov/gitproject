import pickle
from sklearn.linear_model import LinearRegression

class Model():
    '''
     Methods:
         __init__: initialization of model
         train : fit linear model with x, y
         predict: make predictions for x after training
         save: save linear trained model
         load : load linear trained model

     '''
    def __init__(self):
        self.model = LinearRegression()

    def trainee(self,x ,y ):
        self.model.fit(x, y)
        print("model is trained")

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        pickle.dump(self.model , open(path, "wb"))

    def load(self, path):
        return pickle.load(open(path, "rb"))

