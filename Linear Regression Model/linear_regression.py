import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("Housing.csv")

# Features and target
X = df.drop("price", axis=1)
y = df["price"]



# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model =LinearRegression()
# Train the model
model.fit(X_train, y_train)

# Predict on test data and evaluate performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n📉 Mean Squared Error on Test Set: {mse:,.2f}")
print(f"📏 Root Mean Squared Error (RMSE): {rmse:,.2f}")

# -------- Predict new data from user input --------

print("\nPlease enter the following details:")
mainroad = input("mainroad (yes/no): ")
guestroom = input("guestroom (yes/no): ")
basement = input("basement (yes/no): ")
hotwaterheating = input("hotwaterheating (yes/no): ")
airconditioning = input("airconditioning (yes/no): ")
prefarea = input("prefarea (yes/no): ")
furnishingstatus = input("furnishingstatus (furnished/semi-furnished/unfurnished): ")
area = int(input("area (in square feet): "))
bedrooms = int(input("bedrooms: "))
bathrooms = int(input("bathrooms: "))
stories = int(input("stories: "))
parking = int(input("parking: "))

# Prepare the input row as a DataFrame
new_data = pd.DataFrame([{
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus,
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking
}])

# Predict the price
predicted_price = model.predict(new_data)[0]
print(f"\n🏠 Predicted House Price: {int(predicted_price):,} PKR")
