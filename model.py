
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
# Load the dataset
df = pd.read_csv("/Users/vinayakprakash/Documents/files/AgroSense/model/Crop_recommendation.csv")


df = df.drop("rainfall", axis=1)

X = df.drop("label", axis=1)  # features
y = df["label"] 
# target variable

le = LabelEncoder()
y_encoded = le.fit_transform(y) 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize and train the XGBoost classifier
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model, scaler, and label encoder for future use
joblib.dump(model, "/Users/vinayakprakash/Documents/files/AgroSense/model/crop_recommender.pkl")
joblib.dump(scaler, "/Users/vinayakprakash/Documents/files/AgroSense/model/scaler.pkl")
joblib.dump(le, "/Users/vinayakprakash/Documents/files/AgroSense/model/label_encoder.pkl")

print("Model, scaler, and label encoder saved successfully!")