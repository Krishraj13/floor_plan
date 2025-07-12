import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Load dataset
df = pd.read_csv("floor_plan_final.csv")

# Add engineered features
df["area_per_room"] = df.apply(lambda row: row["flat_area_sft"] / row["no_of_rooms"] if row["no_of_rooms"] > 0 else row["flat_area_sft"], axis=1)
df["area_per_flat"] = df["flat_area_sft"] / df["no_of_flats"]
df["room_density"] = df["no_of_rooms"] / df["flat_area_sft"]

# Define features and targets
features = ["floor_number", "no_of_flats", "flat_area_sft", "no_of_rooms", "area_per_room", "area_per_flat", "room_density"]
X = df[features]
y = df[["sand_tons", "cement_tons", "no_of_bricks", "labour_count", "labour_hours"]]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "xgb_feature_scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost model
base_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Trained Successfully with Feature Engineering + XGBoost")
print("-----------------------------------------------------")
print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
print(f"🔹 R² Score (Accuracy): {r2:.4f} (~{r2 * 100:.2f}%)")
print("-----------------------------------------------------")

# Save model
joblib.dump(model, "material_predictor_model.pkl")
print("📦 Model saved as 'material_predictor_model.pkl'")
