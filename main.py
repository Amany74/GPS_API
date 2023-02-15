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