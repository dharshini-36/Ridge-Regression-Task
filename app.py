import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# -----------------------------
# 1. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")
    return df

df = load_data()

st.title("🏠 California Housing Price Prediction")

st.write("### Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# 2. Preprocessing
# -----------------------------
# One-hot encode oceanProximity
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Features & Target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Ridge model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 3. User Input
# -----------------------------
st.sidebar.header("Enter House Details")

longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.number_input("House Age", value=30)
total_rooms = st.sidebar.number_input("Total Rooms", value=2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=400)
population = st.sidebar.number_input("Population", value=1000)
households = st.sidebar.number_input("Households", value=300)
median_income = st.sidebar.number_input("Median Income", value=5.0)

ocean = st.sidebar.selectbox(
    "Ocean Proximity",
    ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
)

# Create input dataframe
input_dict = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encode input
input_df = pd.get_dummies(input_df)

# Align with training columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_df)

# -----------------------------
# 4. Prediction
# -----------------------------
prediction = model.predict(input_scaled)

st.write("## 💰 Predicted House Price")
st.success(f"${prediction[0]:,.2f}")
