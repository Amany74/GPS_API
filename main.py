# main.py
import pickle
import numpy as np
import pandas as pd


from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def hello():
    return {"message":"Hello from PIMA"}

@app.get("/predictknn")
def predict(Year:int,Month:int,Day:int,Hour:int):
    """
    KNN
    paramters : 4 all integers in an array reshaped into (1,4)
    [Year , Month , Day ,Hour]
    """
    model = pickle.load(open('./trained_models/trained_knn_00.pkl','rb'))
    newX = pd.DataFrame((np.asarray([Year,Month,Day,Hour])).reshape(1,4))
    # Values validation
    makeprediction = model.predict(newX)
    return {'lat' : makeprediction[0][0],'lng': makeprediction[0][1]}


@app.get("/predictrf")
def predict(Year:int,Month:int,Day:int,Hour:int):
    """
    Random Forest
    paramters : 4 all integers in an array reshaped into (1,4)
    [Year , Month , Day ,Hour]
    """
    model = pickle.load(open('./trained_models/trained_random_forest_00.pkl','rb'))
    newX = pd.DataFrame((np.asarray([Year,Month,Day,Hour])).reshape(1,4))
    # Values validation
    makeprediction = model.predict(newX)
    return {'lat' : makeprediction[0][0],'lng': makeprediction[0][1]}

@app.get("/predictr")
def predict(Year:int,Month:int,Day:int,Hour:int):
    """Decision Tree
    paramters : 4 all integers in an array reshaped into (1,4)
    [Year , Month , Day ,Hour]
    """
    model = pickle.load(open('./trained_models/trained_regressor_tree_00.pkl','rb'))
    newX = pd.DataFrame((np.asarray([Year,Month,Day,Hour])).reshape(1,4))
    # Values validation
    makeprediction = model.predict(newX)
    return {'lat' : makeprediction[0][0],'lng': makeprediction[0][1]}

@app.get("/predictn")
def predict(Year:int,Month:int,Day:int,Hour:int):
    """LSTM
    paramters : 4 all integers in an array reshaped into (1,4)
    [Year , Month , Day ,Hour]
    """
    model = tf.keras.models.load_model('lstm_model_v1.h5')
    newX = pd.DataFrame((np.asarray(X_new).reshape(1,4)))

    # Values validation
    makeprediction = model.predict(np.expand_dims(X_new, axis=2))
    return {'lat' : makeprediction[0][0],'lng': makeprediction[0][1]}


@app.get("/predict")
def predict(Year:int,Month:int,Day:int,Hour:int):
    """Assembly method combining KNN and RNN
    paramters : 4 all integers in an array reshaped into (1,4)
    [Year , Month , Day ,Hour]
    """
    #Model KNN
    model1 = pickle.load(open('./trained_models/trained_knn_00.pkl', 'rb'))
    newX1 = pd.DataFrame((np.asarray([Year, Month, Day, Hour])).reshape(1, 4))
    # Values validation
    makeprediction1 = model1.predict(newX1)

    #Model LSTM
    model2 = tf.keras.models.load_model('lstm_model_v1.h5')
    newX2 = pd.DataFrame((np.asarray([Year, Month, Day, Hour])).reshape(1, 4))
    newX2 = pd.DataFrame((np.asarray(newX2).reshape(1, 4)))

    # Values validation
    makeprediction2 = model2.predict(np.expand_dims(newX2, axis=2))

    #return values after combination
    lat = (makeprediction1[0][0] + makeprediction2[0][0])/2
    lng = (makeprediction1[0][1] + makeprediction2[0][1])/2
    return {'lat': lat, 'lng': lng}
