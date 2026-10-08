from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI(title="Iris ML API")

# Cargamos el modelo (asegúrate de que el nombre coincida)
model = joblib.load("model.pkl")

@app.get("/")
def read_root():
    return {"message": "API de Iris funcionando. Ve a /docs para probarla."}

@app.get("/predict")
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(X)[0]
    labels = ["setosa", "versicolor", "virginica"]

    return {
        "class_id": int(pred),
        "class_name": labels[pred]
    }