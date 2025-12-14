import pandas as pd
import joblib

# Load saved model, scaler, and label encoder
model = joblib.load("/Users/vinayakprakash/Documents/files/AgroSense/model/crop_recommender.pkl")
scaler = joblib.load("/Users/vinayakprakash/Documents/files/AgroSense/model/scaler.pkl")
le = joblib.load("/Users/vinayakprakash/Documents/files/AgroSense/model/label_encoder.pkl")

# Function to predict crop based on sensor input
def recommend_crop(sensor_data: dict) -> str:
   
    df = pd.DataFrame([sensor_data])
    X_scaled = scaler.transform(df)           # scale features
    y_pred_num = model.predict(X_scaled)      # numeric prediction
    y_pred_label = le.inverse_transform(y_pred_num)  # convert back to crop name
    return y_pred_label[0]

# Example sensor data to test
sensor_input = {
    "N": 50,
    "P": 30,
    "K": 40,
    "temperature": 25,
    "humidity": 60,
    "ph": 6.5
}

# Get prediction
recommended_crop = recommend_crop(sensor_input)
print("Recommended crop:", recommended_crop)
