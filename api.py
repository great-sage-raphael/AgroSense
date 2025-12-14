from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model, scaler, label encoder
model = joblib.load("/Users/vinayakprakash/Documents/files/AgroSense/model/crop_recommender.pkl")
scaler = joblib.load("/Users/vinayakprakash/Documents/files/AgroSense/model/scaler.pkl")
le = joblib.load("/Users/vinayakprakash/Documents/files/AgroSense/model/label_encoder.pkl")

app = FastAPI()

# Define input schema
class SensorData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float

@app.post("/predict")
def predict_crop(data: SensorData):
    df = pd.DataFrame([data.dict()])
    X_scaled = scaler.transform(df)
    y_pred_num = model.predict(X_scaled)
    y_pred_label = le.inverse_transform(y_pred_num)
    return {"recommended_crop": y_pred_label[0]}
